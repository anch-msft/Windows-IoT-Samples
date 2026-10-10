using EdgeAIKiosk.Interfaces;
using EdgeAIKiosk.Models;
using EdgeAIKiosk.Services;
using Microsoft.UI.Xaml;
using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using EdgeAIKiosk.Pipeline;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;

namespace EdgeAIKiosk.Views;

public sealed partial class ShoppingPage : Page
{
    public ObservableCollection<ScannedItem> ScannedItems { get; } = new();
    public Visibility BarcodeInputVisibility =>
        KioskSettings.ScannerMode == ScannerMode.Keyboard ? Visibility.Visible : Visibility.Collapsed;

    private readonly LiveInferenceView _liveInference = new();
    private IBarcodeScanner? _scanner;

    public ShoppingPage()
    {
        InitializeComponent();
        ScannedItemsListBox.ItemsSource = ScannedItems;
        LiveInferenceHost.Content = _liveInference;
    }

    protected override void OnNavigatedFrom(NavigationEventArgs e)
    {
        base.OnNavigatedFrom(e);
        if (_scanner is not null)
        {
            _scanner.BarcodeScanned -= OnBarcodeScanned;
        }
        _scanner?.Dispose();
        _scanner = null;
    }

    protected override async void OnNavigatedTo(NavigationEventArgs e)
    {
        base.OnNavigatedTo(e);
        PayNowButton.IsEnabled = false;
        await StartupTask.Run(async () =>
        {
            _scanner = KioskSettings.ScannerMode == ScannerMode.HidScanner
                ? new HidBarcodeScanner()
                : new KeyboardBarcodeScanner(BarcodeInputBox);
            _scanner.BarcodeScanned += OnBarcodeScanned;
            await _scanner.StartAsync();
            await _liveInference.StartAsync();
            LoadingOverlay.Visibility = Visibility.Collapsed;
            PayNowButton.IsEnabled = true;
        }, "Hardware connection error", message =>
        {
            _scanner?.Dispose();
            _ = _liveInference.StopAsync();
            ShowLoadingError(message);
        });
    }

    private void OnShoppingPageLoaded(object sender, RoutedEventArgs e)
    {
        if (KioskSettings.ScannerMode == ScannerMode.Keyboard)
        {
            BarcodeInputBox.Focus(FocusState.Programmatic);
        }
    }

    /// <summary>
    /// Receives a barcode from any scanner type, adds unique items to the cart, and syncs with live inference.
    /// </summary>
    private void OnBarcodeScanned(string barcode)
    {
        DispatcherQueue.TryEnqueue(() =>
        {
            if (!PayNowButton.IsEnabled || ScannedItems.Any(item => item.Barcode == barcode))
            {
                return;
            }
            ScannedItems.Add(new ScannedItem(barcode, barcode, 1));
            _liveInference.SetScannedItems(ScannedItems);
        });
    }

    private void OnVoidItemClick(object sender, RoutedEventArgs e)
    {
        if (ScannedItemsListBox.SelectedItem is ScannedItem item)
        {
            ScannedItems.Remove(item);
        }
        _liveInference.SetScannedItems(ScannedItems);
    }

    /// <summary>
    /// Pauses live inference, verifies the scanned cart, and opens the result page.
    /// </summary>
    private async void OnPayNowClick(object sender, RoutedEventArgs e)
    {
        PayNowButton.IsEnabled = false;
        LoadingProgressRing.Visibility = Visibility.Visible;
        LoadingTextBlock.Text = "Verifying basket...";
        LoadingOverlay.Visibility = Visibility.Visible;
        try
        {
            await _liveInference.PauseAsync();
            ShoppingVerifier shoppingVerifier = new(
                _liveInference.ImageCapture,
                _liveInference.Preprocessor,
                _liveInference.ModelLoader,
                new MajorityFrames());
            VerificationResult verificationResult = await shoppingVerifier.Verify(ScannedItems.ToList());

            LiveInferenceHost.Content = null;
            Frame.Navigate(typeof(AlertPage), new CheckoutNavigationArgs(verificationResult, _liveInference));
        }
        catch (Exception exception)
        {
            Trace.TraceError(exception.ToString());
            LiveInferenceHost.Content = _liveInference;
            _liveInference.Resume();
            PayNowButton.IsEnabled = true;
            ShowLoadingError($"Verification failed: {exception.Message}");
        }
    }

    private void ShowLoadingError(string message)
    {
        LoadingProgressRing.Visibility = Visibility.Collapsed;
        LoadingTextBlock.Text = message;
        LoadingOverlay.Visibility = Visibility.Visible;
    }
}
