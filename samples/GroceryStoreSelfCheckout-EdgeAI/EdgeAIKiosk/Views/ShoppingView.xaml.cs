using EdgeAI_ObjectDetection.Controls;
using EdgeAIKiosk.Interfaces;
using EdgeAIKiosk.Models;
using EdgeAIKiosk.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace EdgeAIKiosk.Views;

public sealed partial class ShoppingView : Window
{
    public ObservableCollection<ScannedItem> ScannedItems { get; } = new();
    public Visibility BarcodeInputVisibility =>
        _settings.ScannerMode == ScannerMode.Keyboard ? Visibility.Visible : Visibility.Collapsed;

    private readonly KioskSettings _settings;
    private readonly KioskInferenceOptions _options;
    private readonly YoloInferenceView _inferenceView = new()
    {
        PreviewStretch = Stretch.UniformToFill,
        Foreground = new SolidColorBrush(Microsoft.UI.Colors.White),
        ShowConfidence = false,
        ShowInferenceTime = false,
        ShowEndToEndInferenceTime = false
    };
    private IBarcodeScanner? _scanner;
    private IReadOnlyList<Detection>? _latestDetections;
    private bool _cartEnabled;
    private bool _started;
    private bool _closed;
    private bool _transferred;

    internal ShoppingView(KioskSettings settings, KioskInferenceOptions options)
    {
        _settings = settings;
        _options = options;
        InitializeComponent();
        WindowLayout.Maximize(this);
        Closed += OnWindowClosed;
        _inferenceView.Faulted += OnInferenceFaulted;
        _inferenceView.DetectionsUpdated += OnDetectionsUpdated;
    }

    private void OnWindowClosed(object sender, WindowEventArgs args)
    {
        _closed = true;
        _inferenceView.Faulted -= OnInferenceFaulted;
        _inferenceView.DetectionsUpdated -= OnDetectionsUpdated;
        DisposeScanner();
    }

    private void DisposeScanner()
    {
        if (_scanner is null) return;
        _scanner.BarcodeScanned -= OnBarcodeScanned;
        _scanner.Dispose();
        _scanner = null;
    }

    private async void OnShoppingViewLoaded(object sender, RoutedEventArgs e)
    {
        if (_started || _closed) return;
        _started = true;
        try
        {
            ScannedItemsListBox.ItemsSource = ScannedItems;
            LiveInferenceHost.Content = _inferenceView;
            SetCartEnabled(false);
            _scanner = _settings.ScannerMode == ScannerMode.HidScanner
                ? new HidBarcodeScanner()
                : new KeyboardBarcodeScanner(BarcodeInputBox);
            _scanner.BarcodeScanned += OnBarcodeScanned;
            await _scanner.StartAsync();
            if (_closed) return;
            await KioskInference.StartAsync(_inferenceView, _options, CancellationToken.None);
            if (_closed) return;
            SetCartEnabled(true);
            LoadingOverlay.Visibility = Visibility.Collapsed;
        }
        catch (Exception error)
        {
            Trace.TraceError(error.ToString());
            DisposeScanner();
            if (!_closed) ShowLoadingError($"Hardware connection error: {error.Message}");
        }
    }

    private void OnInferenceFaulted(object? sender, InferenceFaultedEventArgs args)
    {
        if (_closed || _transferred) return;
        SetCartEnabled(false);
        ShowLoadingError($"Inference failed: {args.Exception.Message}");
    }

    private void OnDetectionsUpdated(object? sender, DetectionsUpdatedEventArgs args)
    {
        if (_closed || _transferred) return;
        _latestDetections = args.Detections;
        DetectedCountTextBlock.Text = $"Detected: {args.Detections.Count} items";
        PayNowButton.IsEnabled = _cartEnabled;
    }

    private void OnBarcodeScanned(string barcode)
    {
        DispatcherQueue.TryEnqueue(() =>
        {
            if (_closed || _transferred || !_cartEnabled) return;
            if (ScannedItems.Any(item => item.Barcode == barcode)) return;
            ScannedItems.Add(new ScannedItem(barcode, barcode, 1));
        });
    }

    private void OnVoidItemClick(object sender, RoutedEventArgs e)
    {
        if (ScannedItemsListBox.SelectedItem is ScannedItem item)
            ScannedItems.Remove(item);
    }

    private void OnPayNowClick(object sender, RoutedEventArgs e)
    {
        SetCartEnabled(false);
        try
        {
            if (_latestDetections is null)
            {
                throw new InvalidOperationException("Wait for the first inference result before checking out.");
            }

            Detection[] detections = _latestDetections.ToArray();
            VerificationResult result = ShoppingVerifier.Verify(ScannedItems.ToArray(), detections, _options.SelectedClasses);
            AlertWindow alert = new(result, _inferenceView, _settings);
            _inferenceView.Faulted -= OnInferenceFaulted;
            _inferenceView.DetectionsUpdated -= OnDetectionsUpdated;
            LiveInferenceHost.Content = null;
            _transferred = true;
            alert.Activate();
            Close();
        }
        catch (Exception error)
        {
            Trace.TraceError(error.ToString());
            if (_closed) return;
            SetCartEnabled(true);
            if (!_closed) ShowLoadingError($"Verification failed: {error.Message}");
        }
    }

    private void SetCartEnabled(bool enabled)
    {
        _cartEnabled = enabled;
        PayNowButton.IsEnabled = enabled && _latestDetections is not null;
        VoidItemButton.IsEnabled = enabled;
        BarcodeInputBox.IsEnabled = enabled;
    }

    private void ShowLoadingError(string message)
    {
        LoadingProgressRing.Visibility = Visibility.Collapsed;
        LoadingTextBlock.Text = message;
        LoadingOverlay.Visibility = Visibility.Visible;
    }
}
