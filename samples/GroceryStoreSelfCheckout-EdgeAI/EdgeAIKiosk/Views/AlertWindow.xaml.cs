using EdgeAI_ObjectDetection.Controls;
using EdgeAIKiosk.Models;
using EdgeAIKiosk.Services;
using Microsoft.UI.Xaml;
using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace EdgeAIKiosk.Views;

public sealed partial class AlertWindow : Window
{
    private readonly VerificationResult _verificationResult;
    private readonly YoloInferenceView _inferenceView;
    private readonly KioskSettings _settings;
    private Task _startupTask = Task.CompletedTask;
    private bool _started;
    private bool _closed;
    private bool _returningHome;

    internal AlertWindow(VerificationResult result, YoloInferenceView inferenceView,
        KioskSettings settings)
    {
        _verificationResult = result;
        _inferenceView = inferenceView;
        _settings = settings;
        InitializeComponent();
        WindowLayout.Maximize(this);
        Closed += OnWindowClosed;
        _inferenceView.Faulted += OnInferenceFaulted;
        _inferenceView.DetectionsUpdated += OnDetectionsUpdated;
    }

    private async void OnAlertWindowLoaded(object sender, RoutedEventArgs e)
    {
        if (_started || _closed || _returningHome) return;
        _started = true;
        _startupTask = ShowResultAsync();
        await _startupTask;
    }

    private async Task ShowResultAsync()
    {
        try
        {
            if (_verificationResult.IsMatch)
            {
                SuccessPanel.Visibility = Visibility.Visible;
                FailurePanel.Visibility = Visibility.Collapsed;
                await _inferenceView.DisposeAsync();
                return;
            }
            FailurePanel.Visibility = Visibility.Visible;
            SuccessPanel.Visibility = Visibility.Collapsed;
            MismatchedItemsListView.ItemsSource = _verificationResult.Mismatches;
            DetectedCountTextBlock.Text = $"Detected: {_verificationResult.DetectedItems.Count} items";
            LiveInferenceHost.Content = _inferenceView;
        }
        catch (Exception error)
        {
            Trace.TraceError(error.ToString());
            if (!_closed) ShowError($"Verification preview failed: {error.Message}");
        }
    }

    private void OnInferenceFaulted(object? sender, InferenceFaultedEventArgs args)
    {
        if (!_closed) ShowError($"Inference failed: {args.Exception.Message}");
    }

    private void OnDetectionsUpdated(object? sender, DetectionsUpdatedEventArgs args) =>
        DetectedCountTextBlock.Text = $"Detected: {args.Detections.Count} items";

    private async void OnStartAgainClick(object sender, RoutedEventArgs e)
    {
        if (_returningHome || _closed) return;
        _returningHome = true;
        try
        {
            await _startupTask;
            if (_closed) return;
            await _inferenceView.DisposeAsync();
            if (_closed) return;
            new HomeWindow(_settings).Activate();
            Close();
        }
        catch (Exception error)
        {
            Trace.TraceError(error.ToString());
            if (!_closed)
            {
                ShowError($"Could not stop inference: {error.Message}");
                _returningHome = false;
            }
        }
    }

    private void OnWindowClosed(object sender, WindowEventArgs args)
    {
        _closed = true;
        _inferenceView.Faulted -= OnInferenceFaulted;
        _inferenceView.DetectionsUpdated -= OnDetectionsUpdated;
        LiveInferenceHost.Content = null;
    }

    private void ShowError(string message)
    {
        SuccessPanel.Visibility = Visibility.Collapsed;
        FailurePanel.Visibility = Visibility.Visible;
        MismatchedItemsListView.ItemsSource = new[] { message };
    }
}
