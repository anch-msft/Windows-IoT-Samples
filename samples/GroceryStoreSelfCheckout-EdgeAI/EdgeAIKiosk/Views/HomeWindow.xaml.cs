using EdgeAI_ObjectDetection.Controls;
using EdgeAI_ObjectDetection.Pipeline;
using EdgeAIKiosk.Services;
using Microsoft.ML.OnnxRuntime;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.Windows.AI.MachineLearning;
using System;
using System.Collections.Generic;
using System.Collections.Frozen;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Windows.Media.Capture.Frames;

namespace EdgeAIKiosk.Views;

public sealed partial class HomeWindow : Window
{
    private readonly KioskSettings _settings;
    private bool _optionsLoaded;

    public HomeWindow() : this(new KioskSettings())
    {
    }

    internal HomeWindow(KioskSettings settings)
    {
        _settings = settings;
        InitializeComponent();
        WindowLayout.Maximize(this);
    }

    private async void OnHomeWindowLoaded(object sender, RoutedEventArgs e) =>
        await StartupTask.Run(LoadPickerOptions, "Startup setup failed", ShowStartupError);

    private async Task LoadPickerOptions()
    {
        LoadScannerOptions();
        LoadModelOptions();
        await LoadHardwareOptions();
        await LoadCameraOptions();
        _optionsLoaded = true;
        UpdateStartButton();
    }

    private void LoadScannerOptions()
    {
        List<ScannerPickerOption> options = new()
        {
            new("HID Barcode Scanner", ScannerMode.HidScanner),
            new("Keyboard Mode", ScannerMode.Keyboard)
        };
        ScannerPickerComboBox.ItemsSource = options;
        ScannerPickerComboBox.SelectedItem = options.FirstOrDefault(o => o.Value == _settings.ScannerMode) ?? options[0];
    }

    private void LoadModelOptions()
    {
        string modelDirectory = Path.Combine(AppContext.BaseDirectory, "Models");
        List<PickerOption> modelOptions = (Directory.Exists(modelDirectory) ? Directory.GetFiles(modelDirectory, "*.onnx") : [])
            .Select(path => new PickerOption(Path.GetFileName(path), Path.Combine("Models", Path.GetFileName(path))))
            .ToList();
        ModelPickerComboBox.ItemsSource = modelOptions;
        ModelPickerComboBox.SelectedItem = FindSelectedOption(modelOptions, _settings.ModelFileName);
        ModelPickerComboBox.SelectionChanged += (_, _) => UpdateStartButton();
    }

    private static PickerOption? FindSelectedOption(IReadOnlyList<PickerOption> options, string? selectedValue) =>
        options.FirstOrDefault(option => option.Value == selectedValue) ?? options.FirstOrDefault();

    private async Task LoadHardwareOptions()
    {
        try { await ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync(); }
        catch (COMException exception) { Trace.TraceWarning($"Windows ML provider registration failed: {exception.Message}"); }
        List<HardwarePickerOption> options = BuildHardwareOptions(OrtEnv.Instance().GetEpDevices().Select(device => device.HardwareDevice.Type));
        HardwarePickerComboBox.ItemsSource = options;
        HardwarePickerComboBox.SelectedItem = options.FirstOrDefault(option => option.Value == _settings.PreferredHardware) ?? options[0];
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
        IReadOnlyList<MediaFrameSourceGroup> groups = await MediaFrameSourceGroup.FindAllAsync();
        List<CameraPickerOption> cameraOptions = groups.Where(group => group.SourceInfos.Any(YoloPipeline.IsColorVideoSource))
            .Select(group => new CameraPickerOption(group.DisplayName, group)).ToList();
        CameraPickerComboBox.ItemsSource = cameraOptions;
        CameraPickerComboBox.SelectedItem = cameraOptions.FirstOrDefault(option => option.Group.Id == _settings.CameraGroupId)
            ?? cameraOptions.FirstOrDefault();
        CameraPickerComboBox.SelectionChanged += (_, _) => UpdateStartButton();
    }

    private void UpdateStartButton()
    {
        if (!_optionsLoaded) return;
        bool hasModel = ModelPickerComboBox.SelectedItem is PickerOption;
        bool hasCamera = CameraPickerComboBox.SelectedItem is CameraPickerOption;
        StartNowButton.IsEnabled = hasModel && hasCamera;
        StartupErrorTextBlock.Text = !hasModel ? "Copy a compatible ONNX model into the Models folder and restart."
            : !hasCamera ? "No compatible color camera was found. Connect a camera and restart." : string.Empty;
        StartupErrorTextBlock.Visibility = StartNowButton.IsEnabled ? Visibility.Collapsed : Visibility.Visible;
    }

    private void ShowStartupError(string message)
    {
        StartNowButton.IsEnabled = false;
        StartupErrorTextBlock.Text = message;
        StartupErrorTextBlock.Visibility = Visibility.Visible;
    }

    /// <summary>
    /// Saves the selected model and camera, opens the shopping flow, and closes this window.
    /// </summary>
    private void OnStartNowClick(object sender, RoutedEventArgs e)
    {
        _settings.ScannerMode = ((ScannerPickerOption)ScannerPickerComboBox.SelectedItem).Value;
        _settings.ModelFileName = ((PickerOption)ModelPickerComboBox.SelectedItem).Value;
        _settings.PreferredHardware = ((HardwarePickerOption)HardwarePickerComboBox.SelectedItem).Value;
        MediaFrameSourceGroup camera = ((CameraPickerOption)CameraPickerComboBox.SelectedItem).Group;
        _settings.CameraGroupId = camera.Id;
        KioskInferenceOptions options = new(Path.Combine(AppContext.BaseDirectory, _settings.ModelFileName),
            camera, _settings.PreferredHardware, _settings.AcceptedLabels.ToFrozenSet(StringComparer.OrdinalIgnoreCase));
        new ShoppingView(_settings, options).Activate();
        Close();
    }

    private void OnSettingsClick(object sender, RoutedEventArgs e)
    {
        if (SettingsPanel.Visibility == Visibility.Visible) HideSettings();
        else ShowSettings();
    }

    private void OnSettingsDismissLayerTapped(object sender, TappedRoutedEventArgs e) =>
        HideSettings();

    private void ShowSettings()
    {
        SettingsPanel.Visibility = Visibility.Visible;
        SettingsDismissLayer.Visibility = Visibility.Visible;
    }

    private void HideSettings()
    {
        SettingsPanel.Visibility = Visibility.Collapsed;
        SettingsDismissLayer.Visibility = Visibility.Collapsed;
    }

    private void OnEditLabelsClick(object sender, RoutedEventArgs e)
    {
        PopulateLabelCheckboxes();
        LabelDialogOverlay.Visibility = Visibility.Visible;
    }

    private void PopulateLabelCheckboxes()
    {
        LabelsStackPanel.Children.Clear();
        foreach (string[] labelRow in YoloInferenceView.GetSupportedClasses().Chunk(3))
        {
            LabelsStackPanel.Children.Add(BuildLabelRow(labelRow));
        }
    }

    /// <summary>
    /// Builds one three-column row of label filters for the settings dialog.
    /// </summary>
    private StackPanel BuildLabelRow(IEnumerable<string> labels)
    {
        StackPanel row = new();
        row.Orientation = Orientation.Horizontal;
        row.Spacing = 8;
        foreach (string label in labels) row.Children.Add(BuildLabelCheckBox(label));
        return row;
    }

    private CheckBox BuildLabelCheckBox(string label)
    {
        CheckBox checkBox = new();
        checkBox.Content = label;
        checkBox.Tag = label;
        checkBox.Width = 180;
        checkBox.IsChecked = _settings.AcceptedLabels.Contains(label);
        checkBox.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Microsoft.UI.Colors.White);
        return checkBox;
    }

    private void OnCancelLabelsClick(object sender, RoutedEventArgs e) =>
        LabelDialogOverlay.Visibility = Visibility.Collapsed;

    /// <summary>
    /// Persists the checked labels, then closes the dialog.
    /// </summary>
    private void OnSaveLabelsClick(object sender, RoutedEventArgs e)
    {
        IEnumerable<string> selectedLabels = LabelsStackPanel.Children.OfType<StackPanel>().SelectMany(LabelCheckboxesInRow).Where(IsChecked).Select(LabelFromCheckBox);
        _settings.AcceptedLabels = new HashSet<string>(selectedLabels, StringComparer.OrdinalIgnoreCase);
        LabelDialogOverlay.Visibility = Visibility.Collapsed;
    }

    private static IEnumerable<CheckBox> LabelCheckboxesInRow(StackPanel row) =>
        row.Children.OfType<CheckBox>();

    private static bool IsChecked(CheckBox checkBox) =>
        checkBox.IsChecked == true;

    private static string LabelFromCheckBox(CheckBox checkBox) =>
        (string)checkBox.Tag;

    private sealed record PickerOption(string Name, string Value);
    private sealed record CameraPickerOption(string Name, MediaFrameSourceGroup Group);
    internal sealed record HardwarePickerOption(string Name, OrtHardwareDeviceType? Value);
    private sealed record ScannerPickerOption(string Name, ScannerMode Value);
}
