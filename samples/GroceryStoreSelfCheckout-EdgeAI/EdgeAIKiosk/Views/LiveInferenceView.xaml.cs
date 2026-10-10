using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using EdgeAIKiosk.Interfaces;
using EdgeAIKiosk.Models;
using EdgeAIKiosk.Pipeline;
using EdgeAIKiosk.Services;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.Windows.AI.MachineLearning;
using Rectangle = Microsoft.UI.Xaml.Shapes.Rectangle;
using Windows.Media.Playback;
using Windows.Media.Core;
using Windows.Graphics.Imaging;

namespace EdgeAIKiosk.Views;

public sealed partial class LiveInferenceView : UserControl
{
    private const float UiMinConfidence = 0.5f;
    private readonly HashSet<string> _scannedLabels = new(StringComparer.OrdinalIgnoreCase);
    private IReadOnlyList<DetectedItem> _liveDetections = [];
    private MediaPlayer? _previewPlayer;
    private Yolo26SnapdragonXLoader? _modelLoader;
    private Task? _inferenceTask;
    private bool _isRunning;
    private int _frameWidth = 640;
    private int _frameHeight = 640;

    public ImageCapture ImageCapture { get; } = new();
    public IModelPreprocessor Preprocessor { get; } = new Yolo26Preprocessor();
    public IModelLoader ModelLoader
    {
        get
        {
            if (_modelLoader is null)
            {
                throw new InvalidOperationException("The model has not loaded.");
            }
            return _modelLoader;
        }
    }

    public LiveInferenceView()
    {
        InitializeComponent();
        PoweredByModelTextBlock.Text = $"Powered by {Path.GetFileNameWithoutExtension(KioskSettings.ModelFileName)} model";
    }

    /// <summary>
    /// Starts the camera preview, loads the configured model, and begins live inference.
    /// </summary>
    public async Task StartAsync()
    {
        try
        {
            await ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync();
        }
        catch (COMException exception)
        {
            Trace.TraceWarning($"Windows ML provider registration failed: {exception.Message}");
        }

        MediaSource previewSource = await ImageCapture.StartPreview();
        _previewPlayer = new MediaPlayer { Source = previewSource, RealTimePlayback = true };
        CameraPreview.SetMediaPlayer(_previewPlayer);
        _previewPlayer.Play();

        DetectedCountTextBlock.Text = "Detected: starting inference";
        string modelPath = Path.Combine(AppContext.BaseDirectory, KioskSettings.ModelFileName);
        _modelLoader = await Task.Run(() => new Yolo26SnapdragonXLoader(modelPath, KioskSettings.PreferredHardware));
        PoweredByModelTextBlock.Text = $"Powered by {Path.GetFileNameWithoutExtension(KioskSettings.ModelFileName)} model on {_modelLoader.HardwareDevice}";
        Resume();
    }

    public void Resume()
    {
        if (_isRunning || !ImageCapture.IsInitialized || _modelLoader is null)
        {
            return;
        }
        _isRunning = true;
        _inferenceTask = RunLiveInferenceLoop();
    }

    /// <summary>
    /// Captures frames until paused, updating detections roughly four times per second.
    /// </summary>
    private async Task RunLiveInferenceLoop()
    {
        try
        {
            while (_isRunning && ImageCapture.IsInitialized)
            {
                SoftwareBitmap? frame = await ImageCapture.CaptureFrame();
                if (frame is null)
                {
                    DetectedCountTextBlock.Text = "Detected: no frame";
                }
                else
                {
                    await UpdateDetections(frame);
                }
                await Task.Delay(250);
            }
        }
        finally
        {
            _isRunning = false;
        }
    }

    /// <summary>
    /// Runs inference for one frame, filters low-confidence or irrelevant results, and refreshes the overlay.
    /// </summary>
    private async Task UpdateDetections(SoftwareBitmap frame)
    {
        (_frameWidth, _frameHeight) = (frame.PixelWidth, frame.PixelHeight);
        ModelOutput output = await ModelLoader.RunInference(Preprocessor.Preprocess(frame));
        _liveDetections = output.Detections
            .Where(detection => detection.Confidence >= UiMinConfidence && KioskSettings.AcceptedLabels.Contains(detection.Label))
            .ToList();
        DetectedCountTextBlock.Text = $"Detected: {_liveDetections.Count} items";
        RedrawBoundingBoxes();
    }

    /// <summary>
    /// Rebuilds the detection overlay using aspect-fill scaling for the current preview size.
    /// </summary>
    private void RedrawBoundingBoxes()
    {
        BoundingBoxCanvas.Children.Clear();
        var scale = Math.Max(BoundingBoxCanvas.ActualWidth / _frameWidth, BoundingBoxCanvas.ActualHeight / _frameHeight);
        var offsetX = (BoundingBoxCanvas.ActualWidth - (_frameWidth * scale)) / 2;
        var offsetY = (BoundingBoxCanvas.ActualHeight - (_frameHeight * scale)) / 2;
        foreach (var detection in _liveDetections) DrawDetection(detection, scale, offsetX, offsetY);
    }

    /// <summary>
    /// Draws one labeled box, colored green when its label is in the scanned cart and red otherwise.
    /// </summary>
    private void DrawDetection(DetectedItem detection, double scale, double offsetX, double offsetY)
    {
        var color = IsScanned(detection.Label, _scannedLabels) ? Colors.Green : Colors.Red;
        var rectangle = new Rectangle
        {
            Width = detection.Box.Width * scale,
            Height = detection.Box.Height * scale,
            Stroke = new SolidColorBrush(color),
            StrokeThickness = 4
        };
        var label = new TextBlock
        {
            Text = detection.Label,
            Foreground = new SolidColorBrush(color),
            FontSize = 12,
            FontWeight = Microsoft.UI.Text.FontWeights.Bold
        };
        var x = (detection.Box.X * scale) + offsetX;
        var y = (detection.Box.Y * scale) + offsetY;
        Canvas.SetLeft(rectangle, x);
        Canvas.SetTop(rectangle, y);
        Canvas.SetLeft(label, x);
        Canvas.SetTop(label, y - 16);
        BoundingBoxCanvas.Children.Add(rectangle);
        BoundingBoxCanvas.Children.Add(label);
    }

    public static bool IsScanned(string label, IReadOnlySet<string> scannedLabels) =>
        scannedLabels.Contains(label);

    /// <summary>
    /// Updates which detected labels are represented in the scanned cart.
    /// </summary>
    public void SetScannedItems(IEnumerable<ScannedItem> items)
    {
        _scannedLabels.Clear();
        _scannedLabels.UnionWith(items.Select(item => item.Name).Where(KioskSettings.AcceptedLabels.Contains));
        RedrawBoundingBoxes();
    }

    private void OnBoundingBoxCanvasSizeChanged(object sender, SizeChangedEventArgs e) =>
        RedrawBoundingBoxes();

    public async Task PauseAsync()
    {
        _isRunning = false;
        if (_inferenceTask is not null)
        {
            await _inferenceTask;
        }
    }

    /// <summary>
    /// Stops inference and releases the preview player, model loader, and camera.
    /// </summary>
    public async Task StopAsync()
    {
        await PauseAsync();
        CameraPreview.SetMediaPlayer(null);
        _previewPlayer?.Dispose();
        _previewPlayer = null;
        _modelLoader?.Dispose();
        _modelLoader = null;
        if (ImageCapture.IsInitialized)
        {
            ImageCapture.Stop();
        }
    }
}
