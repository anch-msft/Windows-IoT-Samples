using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using EdgeAI_ObjectDetection.Pipeline;
using Microsoft.ML.OnnxRuntime;
using Microsoft.UI;
using Microsoft.Windows.AI.MachineLearning;
using Microsoft.Windows.Storage.Pickers;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.Core;
using Windows.Media.MediaProperties;
using Windows.Graphics.Imaging;
using Windows.Storage;
using XamlRectangle = Microsoft.UI.Xaml.Shapes.Rectangle;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace EdgeAI_ObjectDetection
{
    /// <summary>
    /// An empty window that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainWindow : Window
    {
        public ObservableCollection<StorageFile> Models { get; } = new();
        public ObservableCollection<MediaFrameSourceGroup> Cameras { get; } = new();

        private StorageFolder? _selectedFolder;
        private StorageFile? _selectedModel;
        private MediaCapture? _mediaCapture;
        private MediaFrameReader? _mediaFrameReader;
        private OrtEpDevice? _selectedExecutionProvider;
        private YoloInferenceEngine? _inferenceEngine;
        private readonly YoloPreprocessor _preprocessor = new();
        private CancellationTokenSource? _inferenceCancellation;
        private Task? _inferenceLoopTask;
        private string? _lastInferenceError;

        public MainWindow()
        {
            InitializeComponent();

            // AppWindow.Resize uses physical pixels, so scale the desired logical size by the window DPI.
            IntPtr hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
            double scale = GetDpiForWindow(hwnd) / 96.0;
            this.AppWindow.Resize(new Windows.Graphics.SizeInt32((int)(1200 * scale), (int)(800 * scale)));
        }

        [DllImport("user32.dll")]
        private static extern uint GetDpiForWindow(IntPtr hwnd);

        private async void MainGrid_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                await ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync();
            }
            catch (COMException exception)
            {
                Trace.TraceWarning($"Windows ML provider registration failed: {exception.Message}");
            }

            OrtEpDevice[] devices = OrtEnv.Instance().GetEpDevices().ToArray();
            foreach (OrtEpDevice device in devices)
            {
                ExecutionProviderComboBox.Items.Add(new ComboBoxItem
                {
                    Content = $"{device.EpName} ({device.HardwareDevice.Type})",
                    Tag = device
                });
            }
            ExecutionProviderComboBox.SelectedIndex = devices.Length > 0 ? 0 : -1;
            ExecutionProviderComboBox.IsEnabled = devices.Length > 0;

            MediaFrameSourceGroup[] cameras = (await MediaFrameSourceGroup.FindAllAsync()).ToArray();

            foreach (MediaFrameSourceGroup camera in cameras.Where(group => group.SourceInfos.Any(IsColorVideoSource)))
                Cameras.Add(camera);

            CameraComboBox.SelectedIndex = Cameras.Count > 0 ? 0 : -1;
            CameraComboBox.IsEnabled = Cameras.Count > 0;

            if (Cameras.Count == 0)
                InferenceStatusTextBlock.Text = "Inference: no compatible color camera was found.";
        }

        private async void ModelFolderButton_Click(object sender, RoutedEventArgs e)
        {
            FolderPicker picker = new FolderPicker(this.AppWindow.Id);
            PickFolderResult? result = await picker.PickSingleFolderAsync();
            if (result is null) return;

            StorageFolder folder = await StorageFolder.GetFolderFromPathAsync(result.Path);
            List<StorageFile> modelFiles = (await folder.GetFilesAsync())
                .Where(file => file.FileType.Equals(".onnx", StringComparison.OrdinalIgnoreCase))
                .OrderBy(file => file.Name, StringComparer.OrdinalIgnoreCase)
                .ToList();

            _selectedFolder = folder;
            _selectedModel = null;
            Models.Clear();
            foreach (StorageFile modelFile in modelFiles)
                Models.Add(modelFile);

            if (modelFiles.Count == 0)
            {
                ModelFolderStatusTextBlock.Text = $"No .onnx model files found in {result.Path}";
                ModelComboBox.SelectedIndex = -1;
            }
            else
            {
                ModelFolderStatusTextBlock.Text = result.Path;
                ModelComboBox.SelectedIndex = 0;
            }
        }

        private async void ModelComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            _selectedModel = ModelComboBox.SelectedItem as StorageFile;
            await ReloadModelAsync();
        }

        private async void ExecutionProviderComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            _selectedExecutionProvider = (ExecutionProviderComboBox.SelectedItem as ComboBoxItem)?.Tag as OrtEpDevice;
            await ReloadModelAsync();
        }

        private async Task ReloadModelAsync()
        {
            SetModelControlsEnabled(false);
            StorageFile? model = _selectedModel;
            OrtEpDevice? device = _selectedExecutionProvider;
            try
            {
                await StopInferenceLoopAsync();

                _lastInferenceError = null;
                DetectionOverlayCanvas.Children.Clear();
                InferenceStatusTextBlock.Text = "Inference: waiting for a model and camera frame.";

                _inferenceEngine?.Dispose();
                _inferenceEngine = null;

                if (model is null)
                {
                    ModelInitializationStatusTextBlock.Text = "Model engine: No model selected.";
                    return;
                }

                if (device is null)
                {
                    ModelInitializationStatusTextBlock.Text = "Model engine: No execution provider is available.";
                    return;
                }

                ModelInitializationStatusTextBlock.Text = $"Model engine: Loading {model.Name}...";
                try
                {
                    YoloInferenceEngine engine = await Task.Run(() => YoloInferenceEngine.Create(model.Path, device));
                    _inferenceEngine = engine;
                    ModelInitializationStatusTextBlock.Text =
                        $"Model engine: Ready ({model.Name}, {engine.HardwareDevice}/{engine.ExecutionProvider}).";
                    StartInferenceLoopIfCameraRunning();
                }
                catch (Exception exception) when (exception is OnnxRuntimeException or IOException or UnauthorizedAccessException
                    or DllNotFoundException or EntryPointNotFoundException or BadImageFormatException or InvalidOperationException)
                {
                    Trace.TraceError(exception.ToString());
                    ModelInitializationStatusTextBlock.Text = $"Model engine: Failed to load {model.Name}: {exception.Message}";
                }
            }
            finally
            {
                SetModelControlsEnabled(true);
            }
        }

        private void SetModelControlsEnabled(bool enabled)
        {
            ModelFolderButton.IsEnabled = enabled;
            ModelComboBox.IsEnabled = enabled && Models.Count > 0;
            ExecutionProviderComboBox.IsEnabled = enabled && ExecutionProviderComboBox.Items.Count > 0;
        }

        private async void CameraComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (CameraComboBox.SelectedItem is not MediaFrameSourceGroup selectedGroup)
                return;

            try
            {
                await StartCameraPreview(selectedGroup);
            }
            catch (Exception exception)
            {
                Trace.TraceWarning($"Camera preview failed: {exception}");
                InferenceStatusTextBlock.Text = $"Inference: camera preview unavailable ({exception.Message})";
                await StopCameraPreviewAsync();
            }
        }

        private static bool IsColorVideoSource(MediaFrameSourceInfo info)
        {
            return info.SourceKind == MediaFrameSourceKind.Color &&
                (info.MediaStreamType == MediaStreamType.VideoPreview ||
                 info.MediaStreamType == MediaStreamType.VideoRecord);
        }

        private async Task StartCameraPreview(MediaFrameSourceGroup sourceGroup)
        {
            CameraComboBox.IsEnabled = false;
            MediaCapture? mediaCapture = null;
            MediaFrameReader? frameReader = null;
            bool frameReaderStarted = false;
            try
            {
                await StopCameraPreviewAsync();

                mediaCapture = new MediaCapture();
                await mediaCapture.InitializeAsync(new MediaCaptureInitializationSettings
                {
                    SourceGroup = sourceGroup,
                    SharingMode = MediaCaptureSharingMode.SharedReadOnly,
                    StreamingCaptureMode = StreamingCaptureMode.Video,
                    MemoryPreference = MediaCaptureMemoryPreference.Cpu
                });

                MediaFrameSource? frameSource = mediaCapture.FrameSources.Values
                    .Where(source => IsColorVideoSource(source.Info))
                    .OrderBy(source => source.Info.MediaStreamType == MediaStreamType.VideoPreview ? 0 : 1)
                    .FirstOrDefault();

                frameSource ??= mediaCapture.FrameSources.Values
                    .FirstOrDefault(source => source.Info.SourceKind == MediaFrameSourceKind.Color);

                if (frameSource is null)
                    throw new InvalidOperationException("The selected camera has no supported color frame source.");

                frameReader = await mediaCapture.CreateFrameReaderAsync(
                    frameSource,
                    MediaEncodingSubtypes.Bgra8);
                MediaFrameReaderStartStatus startStatus = await frameReader.StartAsync();
                if (startStatus != MediaFrameReaderStartStatus.Success)
                    throw new InvalidOperationException($"Could not start the camera frame reader: {startStatus}.");
                frameReaderStarted = true;

                CameraPreview.Source = MediaSource.CreateFromMediaFrameSource(frameSource);
                _mediaCapture = mediaCapture;
                mediaCapture = null;
                _mediaFrameReader = frameReader;
                frameReader = null;
                StartInferenceLoopIfCameraRunning();
            }
            finally
            {
                if (frameReader is not null)
                {
                    if (frameReaderStarted)
                        await frameReader.StopAsync();
                    frameReader.Dispose();
                }
                mediaCapture?.Dispose();
                CameraComboBox.IsEnabled = Cameras.Count > 0;
            }
        }

        private void StartInferenceLoopIfCameraRunning()
        {
            if (_mediaFrameReader is null || _inferenceEngine is null || _inferenceCancellation is not null)
                return;

            _inferenceCancellation = new CancellationTokenSource();
            _inferenceLoopTask = RunInferenceLoopAsync(_inferenceCancellation.Token);
        }

        private async Task StopInferenceLoopAsync()
        {
            CancellationTokenSource? cancellation = _inferenceCancellation;
            Task? loopTask = _inferenceLoopTask;
            if (cancellation is null)
                return;

            cancellation.Cancel();
            if (loopTask is not null)
                await loopTask;

            if (ReferenceEquals(_inferenceCancellation, cancellation))
            {
                _inferenceCancellation = null;
                _inferenceLoopTask = null;
                cancellation.Dispose();
            }
        }

        private async Task StopCameraPreviewAsync()
        {
            await StopInferenceLoopAsync();

            MediaFrameReader? frameReader = _mediaFrameReader;
            MediaCapture? mediaCapture = _mediaCapture;
            _mediaFrameReader = null;
            _mediaCapture = null;

            CameraPreview.Source = null;

            if (frameReader is not null)
            {
                await frameReader.StopAsync();
                frameReader.Dispose();
            }

            mediaCapture?.Dispose();
            DetectionOverlayCanvas.Children.Clear();
        }

        private async Task RunInferenceLoopAsync(CancellationToken cancellationToken)
        {
            YoloModelInput? lastInput = null;
            try
            {
                while (true)
                {
                    if (_inferenceEngine is null)
                    {
                        await Task.Delay(100, cancellationToken);
                        continue;
                    }

                    try
                    {
                        Stopwatch endToEndTimer = Stopwatch.StartNew();
                        MediaFrameReader? frameReader = _mediaFrameReader;
                        if (frameReader is null)
                        {
                            await Task.Delay(100, cancellationToken);
                            continue;
                        }

                        YoloModelInput? acquiredInput = await Task.Run(
                            () => AcquireModelInput(frameReader, _preprocessor),
                            cancellationToken);
                        if (acquiredInput is null)
                        {
                            if (lastInput is null)
                            {
                                DetectionOverlayCanvas.Children.Clear();
                                InferenceStatusTextBlock.Text = "Inference: no camera frame available.";
                                await Task.Delay(10, cancellationToken);
                                continue;
                            }
                        }
                        else
                            lastInput = acquiredInput;

                        Stopwatch inferenceTimer = Stopwatch.StartNew();
                        IReadOnlyList<YoloDetection>? detections =
                            await RunInferenceAsync(lastInput, cancellationToken);
                        TimeSpan inferenceTime = inferenceTimer.Elapsed;
                        if (detections is not null && !cancellationToken.IsCancellationRequested)
                        {
                            _lastInferenceError = null;
                            DrawDetections(detections);
                            TimeSpan endToEndTime = endToEndTimer.Elapsed;
                            InferenceStatusTextBlock.Text =
                                $"Inference: {detections.Count} detections " +
                                $"({inferenceTime.TotalMilliseconds:F0} ms inference, " +
                                $"{endToEndTime.TotalMilliseconds:F0} ms end-to-end)";
                        }
                    }
                    catch (Exception exception) when (exception is OnnxRuntimeException or InvalidOperationException
                        or InvalidCastException or ArgumentException or COMException)
                    {
                        if (cancellationToken.IsCancellationRequested)
                            continue;

                        DetectionOverlayCanvas.Children.Clear();
                        InferenceStatusTextBlock.Text = $"Inference failed: {exception.Message}";
                        if (!StringComparer.Ordinal.Equals(_lastInferenceError, exception.Message))
                            Trace.TraceError(exception.ToString());
                        _lastInferenceError = exception.Message;
                    }
                }
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
            }
        }

        private static YoloModelInput? AcquireModelInput(
            MediaFrameReader frameReader,
            YoloPreprocessor preprocessor)
        {
            using MediaFrameReference? frame = frameReader.TryAcquireLatestFrame();
            SoftwareBitmap? bitmap = frame?.VideoMediaFrame?.SoftwareBitmap;
            return bitmap is null ? null : preprocessor.Preprocess(bitmap);
        }

        private async Task<IReadOnlyList<YoloDetection>?> RunInferenceAsync(
            YoloModelInput input,
            CancellationToken cancellationToken)
        {
            YoloInferenceEngine? engine = _inferenceEngine;
            if (engine is null)
                return null;

            return await Task.Run(() => engine.Run(input), cancellationToken);
        }

        private void DrawDetections(IReadOnlyList<YoloDetection> detections)
        {
            DetectionOverlayCanvas.Children.Clear();
            SolidColorBrush brush = new(Colors.LimeGreen);
            foreach (YoloDetection detection in detections)
            {
                XamlRectangle rectangle = new()
                {
                    Width = detection.Width,
                    Height = detection.Height,
                    Stroke = brush,
                    StrokeThickness = 2
                };
                Canvas.SetLeft(rectangle, detection.X);
                Canvas.SetTop(rectangle, detection.Y);
                DetectionOverlayCanvas.Children.Add(rectangle);

                TextBlock label = new()
                {
                    Text = $"{detection.Label} {detection.Confidence:P0}",
                    Foreground = brush,
                    FontSize = 14,
                    FontWeight = Microsoft.UI.Text.FontWeights.Bold
                };
                Canvas.SetLeft(label, detection.X);
                Canvas.SetTop(label, Math.Max(0, detection.Y - 20));
                DetectionOverlayCanvas.Children.Add(label);
            }
        }

    }
}
