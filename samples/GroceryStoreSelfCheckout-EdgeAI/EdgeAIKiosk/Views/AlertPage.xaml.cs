using System;
using EdgeAIKiosk.Models;
using EdgeAIKiosk.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;

namespace EdgeAIKiosk.Views;

internal sealed record CheckoutNavigationArgs(VerificationResult Result, LiveInferenceView Inference);

public sealed partial class AlertPage : Page
{
    private CheckoutNavigationArgs? _checkout;

    public AlertPage()
    {
        InitializeComponent();
    }

    protected override async void OnNavigatedTo(NavigationEventArgs e)
    {
        base.OnNavigatedTo(e);
        await StartupTask.Run(async () =>
        {
            if (e.Parameter is not CheckoutNavigationArgs checkout)
            {
                throw new ArgumentException("Checkout requires a verification result and its inference session.", nameof(e));
            }

            _checkout = checkout;
            checkout.Inference.SetScannedItems(checkout.Result.ScannedItems);
            SuccessPanel.Visibility = checkout.Result.IsMatch ? Visibility.Visible : Visibility.Collapsed;
            FailurePanel.Visibility = checkout.Result.IsMatch ? Visibility.Collapsed : Visibility.Visible;
            MismatchedItemsListView.ItemsSource = checkout.Result.Mismatches;

            if (checkout.Result.IsMatch)
            {
                await checkout.Inference.StopAsync();
            }
            else
            {
                LiveInferenceHost.Content = checkout.Inference;
            }
        }, "Verification preview failed", message =>
        {
            SuccessPanel.Visibility = Visibility.Collapsed;
            FailurePanel.Visibility = Visibility.Visible;
            MismatchedItemsListView.ItemsSource = new[] { message };
        });
    }

    private void OnAlertPageLoaded(object sender, RoutedEventArgs e)
    {
        if (LiveInferenceHost.Content is LiveInferenceView inference)
        {
            inference.Resume();
        }
    }

    /// <summary>
    /// Stops the shared inference session before returning home.
    /// </summary>
    private async void OnStartAgainClick(object sender, RoutedEventArgs e)
    {
        if (_checkout is not null)
        {
            await _checkout.Inference.StopAsync();
        }
        LiveInferenceHost.Content = null;
        Frame.Navigate(typeof(HomePage));
    }
}
