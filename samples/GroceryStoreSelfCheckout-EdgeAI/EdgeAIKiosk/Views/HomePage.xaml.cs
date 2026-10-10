using EdgeAIKiosk.Services;
using Microsoft.ML.OnnxRuntime;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Navigation;
using Microsoft.Windows.AI.MachineLearning;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Windows.Devices.Enumeration;
using Windows.Media.Devices;

namespace EdgeAIKiosk.Views;

public sealed partial class HomePage : Page
{
    public HomePage()
    {
        InitializeComponent();
    }

    protected override async void OnNavigatedTo(NavigationEventArgs e)
    {
        base.OnNavigatedTo(e);
        await StartupTask.Run(async () =>
        {
            if (KioskSettings.ScannerMode == ScannerMode.Keyboard)
            {
                ScannerPickerComboBox.SelectedItem = KeyboardScannerItem;
            }
            else
            {
                ScannerPickerComboBox.SelectedItem = HidScannerItem;
            }
            LoadModelOptions();
            await LoadHardwareOptions();
            await LoadCameraOptions();
            StartNowButton.IsEnabled = true;
        }, "Startup setup failed", message =>
        {
            StartNowButton.IsEnabled = false;
            StartupErrorTextBlock.Text = message;
            StartupErrorTextBlock.Visibility = Visibility.Visible;
        });
    }

    private void LoadModelOptions()
    {
        List<string> modelNames = Directory.GetFiles(Path.Combine(AppContext.BaseDirectory, "Models"), "*.onnx")
            .Select(Path.GetFileName)
            .OfType<string>()
            .ToList();
        ModelPickerComboBox.ItemsSource = modelNames;
        ModelPickerComboBox.SelectedItem = modelNames
            .FirstOrDefault(name => name == Path.GetFileName(KioskSettings.ModelFileName))
            ?? modelNames.FirstOrDefault();
    }

    private async Task LoadHardwareOptions()
    {
        try
        {
            await ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync();
        }
        catch (COMException exception)
        {
            Trace.TraceWarning($"Windows ML provider registration failed: {exception.Message}");
        }

        List<HardwarePickerOption> options = BuildHardwareOptions(
            OrtEnv.Instance().GetEpDevices().Select(device => device.HardwareDevice.Type));
        HardwarePickerComboBox.ItemsSource = options;
        HardwarePickerComboBox.SelectedItem = options.FirstOrDefault(option => option.Value == KioskSettings.PreferredHardware) ?? options[0];
    }

    internal static List<HardwarePickerOption> BuildHardwareOptions(IEnumerable<OrtHardwareDeviceType> availableHardware)
    {
        HashSet<OrtHardwareDeviceType> available = availableHardware.ToHashSet();
        IEnumerable<HardwarePickerOption> detected = new[] { OrtHardwareDeviceType.CPU, OrtHardwareDeviceType.GPU, OrtHardwareDeviceType.NPU }
            .Where(available.Contains)
            .Select(hardware => new HardwarePickerOption(hardware.ToString(), hardware));
        return [new("Auto", null), .. detected];
    }

    private async Task LoadCameraOptions()
    {
        DeviceInformationCollection devices = await DeviceInformation.FindAllAsync(MediaDevice.GetVideoCaptureSelector());
        CameraPickerComboBox.ItemsSource = devices;
        CameraPickerComboBox.SelectedItem = devices.FirstOrDefault(device => device.Id == KioskSettings.CameraDeviceId)
            ?? devices.FirstOrDefault();
    }

    /// <summary>
    /// Saves the selected model and camera and opens the shopping page.
    /// </summary>
    private void OnStartNowClick(object sender, RoutedEventArgs e)
    {
        if (ScannerPickerComboBox.SelectedItem is ComboBoxItem scanner && scanner == KeyboardScannerItem)
        {
            KioskSettings.ScannerMode = ScannerMode.Keyboard;
        }
        else
        {
            KioskSettings.ScannerMode = ScannerMode.HidScanner;
        }
        KioskSettings.ModelFileName = Path.Combine("Models", (string)ModelPickerComboBox.SelectedItem);
        KioskSettings.PreferredHardware = ((HardwarePickerOption)HardwarePickerComboBox.SelectedItem).Value;
        KioskSettings.CameraDeviceId = ((DeviceInformation?)CameraPickerComboBox.SelectedItem)?.Id;
        Frame.Navigate(typeof(ShoppingPage));
    }

    private void OnSettingsClick(object sender, RoutedEventArgs e)
    {
        if (SettingsPanel.Visibility == Visibility.Visible)
        {
            HideSettings();
        }
        else
        {
            SettingsPanel.Visibility = Visibility.Visible;
            SettingsDismissLayer.Visibility = Visibility.Visible;
        }
    }

    private void OnSettingsDismissLayerTapped(object sender, TappedRoutedEventArgs e)
    {
        HideSettings();
    }

    private void HideSettings()
    {
        SettingsPanel.Visibility = Visibility.Collapsed;
        SettingsDismissLayer.Visibility = Visibility.Collapsed;
    }

    private void OnEditLabelsClick(object sender, RoutedEventArgs e)
    {
        LabelsItemsView.ItemsSource = CocoLabels.Labels;
        LabelsItemsView.DeselectAll();
        for (int index = 0; index < CocoLabels.Labels.Length; index++)
        {
            if (KioskSettings.AcceptedLabels.Contains(CocoLabels.Labels[index]))
            {
                LabelsItemsView.Select(index);
            }
        }
        LabelDialogOverlay.Visibility = Visibility.Visible;
    }

    private void OnCancelLabelsClick(object sender, RoutedEventArgs e)
    {
        LabelDialogOverlay.Visibility = Visibility.Collapsed;
    }

    /// <summary>
    /// Persists the selected labels, then closes the dialog.
    /// </summary>
    private void OnSaveLabelsClick(object sender, RoutedEventArgs e)
    {
        KioskSettings.AcceptedLabels = new HashSet<string>(
            LabelsItemsView.SelectedItems.Cast<string>(), StringComparer.OrdinalIgnoreCase);
        LabelDialogOverlay.Visibility = Visibility.Collapsed;
    }

    internal sealed record HardwarePickerOption(string Name, OrtHardwareDeviceType? Value);
}
