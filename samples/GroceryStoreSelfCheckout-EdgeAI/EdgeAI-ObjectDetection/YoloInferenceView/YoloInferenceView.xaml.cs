using EdgeAI_ObjectDetection.Pipeline;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Windows.Foundation;
using Windows.Media.Core;
using Windows.Media.Playback;
using XamlRectangle = Microsoft.UI.Xaml.Shapes.Rectangle;

namespace EdgeAI_ObjectDetection.Controls;

public sealed partial class YoloInferenceView : UserControl, IAsyncDisposable
{
    private readonly SemaphoreSlim _lifecycle = new(1, 1);
    private CancellationTokenSource? _startupCancellation;
    private CancellationTokenSource? _runCancellation;
    private YoloPipeline? _pipeline;
    private MediaPlayer? _player;
    private MediaSource? _previewSource;
    private Task? _streamTask;
    private InferenceSettings? _settings;
    private bool _disposeRequested;
    private long _geometryVersion;
    private PreviewGeometry? _geometry;
    private IReadOnlyList<Detection> _detections = Array.Empty<Detection>();

    public YoloInferenceState State
    {
        get;
        private set;
    } = YoloInferenceState.Stopped;
    public event EventHandler? StateChanged;
    public event EventHandler<DetectionsUpdatedEventArgs>? DetectionsUpdated;
    public event EventHandler<InferenceFaultedEventArgs>? Faulted;
    public IReadOnlyList<string> GetSupportedClasses() => YoloInferenceEngine.SupportedClasses;

    public static readonly DependencyProperty PreviewStretchProperty = DependencyProperty.Register(
        nameof(PreviewStretch), typeof(Stretch), typeof(YoloInferenceView),
        new PropertyMetadata(Stretch.UniformToFill, OnPreviewStretchChanged));
    public Stretch PreviewStretch
    {
        get => (Stretch)GetValue(PreviewStretchProperty);
        set => SetValue(PreviewStretchProperty, value);
    }

    private static DependencyProperty VisibilityOption(string name) => DependencyProperty.Register(
        name, typeof(bool), typeof(YoloInferenceView), new PropertyMetadata(true, OnPresentationChanged));

    public static readonly DependencyProperty ShowCameraPreviewProperty = VisibilityOption(nameof(ShowCameraPreview));
    public static readonly DependencyProperty ShowBoundingBoxesProperty = VisibilityOption(nameof(ShowBoundingBoxes));
    public static readonly DependencyProperty ShowLabelsProperty = VisibilityOption(nameof(ShowLabels));
    public static readonly DependencyProperty ShowConfidenceProperty = VisibilityOption(nameof(ShowConfidence));
    public static readonly DependencyProperty ShowInferenceStatusProperty = VisibilityOption(nameof(ShowInferenceStatus));
    public static readonly DependencyProperty ShowInferenceTimeProperty = VisibilityOption(nameof(ShowInferenceTime));
    public static readonly DependencyProperty ShowEndToEndInferenceTimeProperty = VisibilityOption(nameof(ShowEndToEndInferenceTime));

    public bool ShowCameraPreview
    {
        get => (bool)GetValue(ShowCameraPreviewProperty); set => SetValue(ShowCameraPreviewProperty, value);
    }
    public bool ShowBoundingBoxes
    {
        get => (bool)GetValue(ShowBoundingBoxesProperty); set => SetValue(ShowBoundingBoxesProperty, value);
    }
    public bool ShowLabels
    {
        get => (bool)GetValue(ShowLabelsProperty); set => SetValue(ShowLabelsProperty, value);
    }
    public bool ShowConfidence
    {
        get => (bool)GetValue(ShowConfidenceProperty); set => SetValue(ShowConfidenceProperty, value);
    }
    public bool ShowInferenceStatus
    {
        get => (bool)GetValue(ShowInferenceStatusProperty); set => SetValue(ShowInferenceStatusProperty, value);
    }
    public bool ShowInferenceTime
    {
        get => (bool)GetValue(ShowInferenceTimeProperty); set => SetValue(ShowInferenceTimeProperty, value);
    }
    public bool ShowEndToEndInferenceTime
    {
        get => (bool)GetValue(ShowEndToEndInferenceTimeProperty);
        set => SetValue(ShowEndToEndInferenceTimeProperty, value);
    }

    public YoloInferenceView()
    {
        InitializeComponent();
        ApplyPresentation();
        Loaded += (_, _) =>
        {
            if (!_disposeRequested)
            {
                RefreshGeometry();
            }
        };
    }

    private static void OnPreviewStretchChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
    {
        var view = (YoloInferenceView)sender;
        if (view._restoringStretch)
        {
            return;
        }

        Stretch value = (Stretch)args.NewValue;
        if (value is not (Stretch.None or Stretch.Uniform or Stretch.UniformToFill) ||
            (view.State != YoloInferenceState.Stopped && value != (Stretch)args.OldValue))
        {
            // Restore without recursively treating restoration as a new runtime change.
            view._restoringStretch = true;
            try
            {
                view.SetValue(PreviewStretchProperty, args.OldValue);
            }
            finally
            {
                view._restoringStretch = false;
            }
            throw new InvalidOperationException("Use None, Uniform, or UniformToFill; change Stretch only while stopped.");
        }
        view.InvalidateGeometry();
    }

    private bool _restoringStretch;

    private static void OnPresentationChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args) =>
        ((YoloInferenceView)sender).ApplyPresentation();

    private void ApplyPresentation()
    {
        CameraPreview.Opacity = ShowCameraPreview ? 1 : 0;
        StillFrame.Opacity = ShowCameraPreview ? 1 : 0;
        StatusText.Visibility = ShowInferenceStatus ? Visibility.Visible : Visibility.Collapsed;
        TimingText.Visibility = ShowInferenceTime ? Visibility.Visible : Visibility.Collapsed;
        EndToEndTimingText.Visibility = ShowEndToEndInferenceTime ? Visibility.Visible : Visibility.Collapsed;
        DrawDetections();
    }

    private void CheckThread()
    {
        if (!DispatcherQueue.HasThreadAccess)
        {
            throw new InvalidOperationException("Call YoloInferenceView lifecycle methods on the UI thread.");
        }
    }

    public async Task StartAsync(InferenceSettings settings, CancellationToken cancellationToken = default)
    {
        CheckThread();
        ArgumentNullException.ThrowIfNull(settings);
        settings.Validate();
        await _lifecycle.WaitAsync(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(_disposeRequested, this);
            if (State != YoloInferenceState.Stopped)
            {
                throw new InvalidOperationException("The control must be stopped before calling StartAsync.");
            }

            _settings = settings;
            _startupCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            SetState(YoloInferenceState.Starting);
            StatusText.Text = $"Loading {Path.GetFileName(settings.ModelPath)} on {settings.ExecutionProvider.EpName}...";
            try
            {
                _pipeline = new YoloPipeline();
                _pipeline.Faulted += Pipeline_Faulted;
                await _pipeline.StartAsync(settings, _startupCancellation.Token);
                _startupCancellation.Token.ThrowIfCancellationRequested();
                if (settings.InferenceType == InferenceType.Stream)
                {
                    _previewSource = MediaSource.CreateFromMediaFrameSource(_pipeline.FrameSource!);
                    _player = new MediaPlayer
                    {
                        RealTimePlayback = true,
                        Source = _previewSource
                    };
                    _player.MediaFailed += Player_MediaFailed;
                    CameraPreview.SetMediaPlayer(_player);
                    _player.Play();
                    CameraPreview.Visibility = Visibility.Visible;
                }
                else
                {
                    CameraPreview.Visibility = Visibility.Collapsed;
                }
                _runCancellation = new CancellationTokenSource();
                SetState(YoloInferenceState.Running);
                StatusText.Text = $"{(settings.InferenceType == InferenceType.Stream ? "Running" : "Ready for one shot")}: " +
                    $"{Path.GetFileName(settings.ModelPath)} ({_pipeline.ProviderDescription})";
                RefreshGeometry();
                if (settings.InferenceType == InferenceType.Stream)
                {
                    _streamTask = RunStreamAsync(_runCancellation.Token);
                }
            }
            catch (Exception error)
            {
                Trace.TraceError(error.ToString());
                try
                {
                    await CleanupAsync();
                }
                catch (Exception cleanupError)
                {
                    Trace.TraceError(cleanupError.ToString());
                    throw new AggregateException("Startup and cleanup both failed.", error, cleanupError);
                }
                finally
                {
                    SetState(YoloInferenceState.Stopped);
                    StatusText.Text = $"Start failed: {error.Message}";
                }
                throw;
            }
            finally
            {
                _startupCancellation.Dispose();
                _startupCancellation = null;
            }
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    public Task StopAsync()
    {
        CheckThread();
        _startupCancellation?.Cancel();
        _runCancellation?.Cancel();
        return StopOrDisposeAsync(false);
    }

    public ValueTask DisposeAsync()
    {
        CheckThread();
        _disposeRequested = true;
        _startupCancellation?.Cancel();
        _runCancellation?.Cancel();
        return new ValueTask(StopOrDisposeAsync(true));
    }

    private async Task StopOrDisposeAsync(bool dispose)
    {
        await _lifecycle.WaitAsync();
        try
        {
            if (State == YoloInferenceState.Disposed)
            {
                return;
            }

            SetState(YoloInferenceState.Stopping);
            try
            {
                await CleanupAsync();
                StatusText.Text = dispose ? "Disposed." : "Not running.";
            }
            catch (Exception error)
            {
                Trace.TraceError(error.ToString());
                StatusText.Text = $"Cleanup failed: {error.Message}";
                throw;
            }
            finally
            {
                SetState(dispose ? YoloInferenceState.Disposed : YoloInferenceState.Stopped);
            }
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private async Task CleanupAsync()
    {
        _runCancellation?.Cancel();
        List<Exception> errors = new();
        void Release(Action action)
        {
            try
            {
                action();
            }
            catch (Exception error)
            {
                errors.Add(error);
            }
        }
        if (_player is { } player)
        {
            player.MediaFailed -= Player_MediaFailed;
            Release(player.Pause);
            Release(() => player.Source = null);
            Release(() => CameraPreview.SetMediaPlayer(null));
            Release(player.Dispose);
        }
        _player = null;
        if (_previewSource is { } source)
        {
            Release(source.Dispose);
        }

        _previewSource = null;
        if (_streamTask is not null)
        {
            try
            {
                await _streamTask;
            }
            catch (Exception error)
            {
                errors.Add(error);
            }
        }
        _streamTask = null;
        try
        {
            if (_pipeline is not null)
            {
                _pipeline.Faulted -= Pipeline_Faulted;
                await _pipeline.DisposeAsync();
            }
        }
        catch (Exception error)
        {
            errors.Add(error);
        }
        finally
        {
            _pipeline = null;
            _runCancellation?.Dispose();
            _runCancellation = null;
            _settings = null;
            StillFrame.Source = null;
            InvalidateGeometry();
            TimingText.Text = string.Empty;
            EndToEndTimingText.Text = string.Empty;
        }
        if (errors.Count > 0)
        {
            throw new AggregateException("Could not completely clean up inference resources.", errors);
        }
    }

    /// <summary>Displays and returns detections and their model-input image. The caller must dispose the snapshot.</summary>
    public async Task<InferenceSnapshot> InferOnceAsync(CancellationToken cancellationToken = default)
    {
        CheckThread();
        await _lifecycle.WaitAsync(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(_disposeRequested, this);
            if (State != YoloInferenceState.Running || _settings?.InferenceType != InferenceType.OneShot)
            {
                throw new InvalidOperationException("Start the control in OneShot mode before requesting inference.");
            }

            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _runCancellation!.Token);
            while (true)
            {
                linked.Token.ThrowIfCancellationRequested();
                PipelineResult? result = await ProcessFrameAsync(linked.Token, captureImage: true);
                if (result is not null)
                {
                    linked.Token.ThrowIfCancellationRequested();
                    var snapshot = new InferenceSnapshot(result.Detections, result.ImagePixels
                        ?? throw new InvalidOperationException("One-shot inference did not capture an image."));
                    try
                    {
                        // Upload before publishing so the still image and boxes change together.
                        var imageSource = new SoftwareBitmapSource();
                        await imageSource.SetBitmapAsync(snapshot.Image);
                        linked.Token.ThrowIfCancellationRequested();
                        StillFrame.Source = imageSource;
                        PresentResult(result);
                        return snapshot;
                    }
                    catch
                    {
                        snapshot.Dispose();
                        throw;
                    }
                }

                await Task.Delay(10, linked.Token);
            }
        }
        catch (Exception error) when (error is not OperationCanceledException and not ObjectDisposedException)
        {
            // Invalid usage is reported to the caller; runtime failures retire the active pipeline.
            if (State == YoloInferenceState.Running && _settings?.InferenceType == InferenceType.OneShot)
            {
                QueueFault(error, _pipeline);
            }

            throw;
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private async Task RunStreamAsync(CancellationToken token)
    {
        YoloPipeline? pipeline = _pipeline;
        try
        {
            while (true)
            {
                token.ThrowIfCancellationRequested();
                Stopwatch cycle = Stopwatch.StartNew();
                await ProcessFrameAsync(token);
                double remaining = 1000 / _settings!.MaxEndToEndFps - cycle.Elapsed.TotalMilliseconds;
                await Task.Delay(TimeSpan.FromMilliseconds(Math.Max(1, remaining)), token);
            }
        }
        catch (OperationCanceledException) when (token.IsCancellationRequested)
        {
        }
        catch (Exception error)
        {
            QueueFault(error, pipeline);
        }
    }

    private async Task<PipelineResult?> ProcessFrameAsync(CancellationToken token, bool captureImage = false)
    {
        RefreshGeometry();
        if (_geometry is not { } geometry)
        {
            return null;
        }

        long version = _geometryVersion;
        var format = _pipeline!.FrameSource!.CurrentFormat.VideoFormat;
        PipelineResult? result = await Task.Run(
            () => _pipeline.TryInfer(geometry, (int)format.Width, (int)format.Height, captureImage, token), token);
        token.ThrowIfCancellationRequested();
        RefreshGeometry();
        if (result is null || version != _geometryVersion)
        {
            return null;
        }

        if (!captureImage)
        {
            PresentResult(result);
        }
        return result;
    }

    private void PresentResult(PipelineResult result)
    {
        _detections = result.Detections;
        DrawDetections();
        TimeSpan endToEndTime = Stopwatch.GetElapsedTime(result.StartedAt);
        TimingText.Text = $"Inference: {result.InferenceTime.TotalMilliseconds:F0} ms";
        EndToEndTimingText.Text = $"E2E inference: {endToEndTime.TotalMilliseconds:F0} ms";
        DetectionsUpdated?.Invoke(this, new DetectionsUpdatedEventArgs(result.Detections));
    }

    private void Player_MediaFailed(MediaPlayer sender, MediaPlayerFailedEventArgs args)
    {
        if (ReferenceEquals(sender, _player))
        {
            QueueFault(new InvalidOperationException($"Camera preview failed: {args.ErrorMessage}", args.ExtendedErrorCode), _pipeline);
        }
    }

    private void Pipeline_Faulted(object? sender, InferenceFaultedEventArgs args) =>
        QueueFault(args.Exception, sender as YoloPipeline);

    private void QueueFault(Exception error, YoloPipeline? failedPipeline)
    {
        Trace.TraceError(error.ToString());
        if (true != DispatcherQueue?.TryEnqueue(async () =>
        {
            if (failedPipeline is null || !ReferenceEquals(_pipeline, failedPipeline))
            {
                return;
            }

            _startupCancellation?.Cancel();
            _runCancellation?.Cancel();
            await _lifecycle.WaitAsync();
            try
            {
                if (failedPipeline is null || !ReferenceEquals(_pipeline, failedPipeline))
                {
                    return;
                }

                SetState(YoloInferenceState.Stopping);
                try
                {
                    await CleanupAsync();
                }
                catch (Exception cleanupError)
                {
                    Trace.TraceError(cleanupError.ToString());
                    error = new AggregateException(error, cleanupError);
                }
                SetState(YoloInferenceState.Stopped);
                StatusText.Text = $"Inference failed: {error.Message}";
            }
            finally
            {
                _lifecycle.Release();
            }
            Faulted?.Invoke(this, new InferenceFaultedEventArgs(error));
        }))
        {
            Trace.TraceError("Could not dispatch the inference fault because the UI dispatcher is shutting down.");
        }
    }

    private void SetState(YoloInferenceState state)
    {
        if (State == state)
        {
            return;
        }

        State = state;
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    private void Viewport_SizeChanged(object sender, SizeChangedEventArgs args)
    {
        InvalidateGeometry();
        RefreshGeometry();
        DrawDetections();
    }

    private void InvalidateGeometry()
    {
        _geometryVersion++;
        _geometry = null;
        if (StillFrame.Source is null)
        {
            _detections = Array.Empty<Detection>();
            Overlay.Children.Clear();
        }
    }

    private void RefreshGeometry()
    {
        double width = Viewport.ActualWidth;
        double height = Viewport.ActualHeight;
        Viewport.Clip = new RectangleGeometry
        {
            Rect = new Rect(0, 0, width, height)
        };
        if (width < 1 || height < 1 || _pipeline?.FrameSource is null)
        {
            if (_geometry is not null)
            {
                InvalidateGeometry();
            }

            return;
        }
        var format = _pipeline.FrameSource.CurrentFormat.VideoFormat;
        int sourceWidth = (int)format.Width;
        int sourceHeight = (int)format.Height;
        if (sourceWidth <= 0 || sourceHeight <= 0)
        {
            return;
        }

        double scale = PreviewStretch switch
        {
            Stretch.Uniform => Math.Min(width / sourceWidth, height / sourceHeight),
            Stretch.UniformToFill => Math.Max(width / sourceWidth, height / sourceHeight),
            Stretch.None => 1 / (XamlRoot?.RasterizationScale ?? 1),
            _ => throw new InvalidOperationException("Fill is not supported.")
        };
        if (Math.Min(width / scale, height / scale) < 2)
        {
            InvalidateGeometry();
            return;
        }
        PreviewGeometry geometry = PreviewGeometry.Create(sourceWidth, sourceHeight, width, height, scale);
        if (_geometry != geometry)
        {
            InvalidateGeometry();
            _geometry = geometry;
        }
        // Explicit placement keeps preview and crop math identical, including None at high DPI.
        CameraPreview.Width = sourceWidth * scale;
        CameraPreview.Height = sourceHeight * scale;
        Canvas.SetLeft(CameraPreview, geometry.ImageX);
        Canvas.SetTop(CameraPreview, geometry.ImageY);
    }

    private void DrawDetections()
    {
        Overlay.Children.Clear();
        PreviewGeometry? displayGeometry = _geometry;
        if (StillFrame.Source is not null)
        {
            if (Viewport.ActualWidth <= 0 || Viewport.ActualHeight <= 0)
            {
                return;
            }
            displayGeometry = PreviewGeometry.FitSquare(YoloInferenceEngine.InputSize,
                Viewport.ActualWidth, Viewport.ActualHeight);
            StillFrame.Width = displayGeometry.Value.OverlaySize;
            StillFrame.Height = displayGeometry.Value.OverlaySize;
            Canvas.SetLeft(StillFrame, displayGeometry.Value.OverlayX);
            Canvas.SetTop(StillFrame, displayGeometry.Value.OverlayY);
        }
        if (displayGeometry is not { } geometry)
        {
            return;
        }
        Overlay.Clip = new RectangleGeometry
        {
            Rect = new Rect(geometry.OverlayX, geometry.OverlayY, geometry.OverlaySize, geometry.OverlaySize)
        };

        SolidColorBrush brush = new(Colors.LimeGreen);
        foreach (Detection detection in _detections)
        {
            DetectionBox box = detection.BoundingBox;
            double x = geometry.OverlayX + box.X * geometry.OverlaySize;
            double y = geometry.OverlayY + box.Y * geometry.OverlaySize;
            if (ShowBoundingBoxes)
            {
                XamlRectangle rectangle = new()
                {
                    Width = box.Width * geometry.OverlaySize,
                    Height = box.Height * geometry.OverlaySize,
                    Stroke = brush,
                    StrokeThickness = 2
                };
                Canvas.SetLeft(rectangle, x);
                Canvas.SetTop(rectangle, y);
                Overlay.Children.Add(rectangle);
            }
            if (ShowLabels || ShowConfidence)
            {
                TextBlock label = new()
                {
                    Text = ShowLabels
                        ? (ShowConfidence ? $"{detection.Label} {detection.Confidence:P0}" : detection.Label)
                        : $"{detection.Confidence:P0}",
                    Foreground = brush,
                    FontSize = 14,
                    MaxWidth = Math.Max(0, geometry.OverlayX + geometry.OverlaySize - x),
                    TextTrimming = TextTrimming.CharacterEllipsis
                };
                Canvas.SetLeft(label, x);
                Canvas.SetTop(label, Math.Max(geometry.OverlayY, y - 20));
                Overlay.Children.Add(label);
            }
        }
    }
}
