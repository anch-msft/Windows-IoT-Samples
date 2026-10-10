using EdgeAIKiosk.Interfaces;
using EdgeAIKiosk.Models;
using EdgeAIKiosk.Pipeline;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using EdgeAIKiosk.Interfaces;
using EdgeAIKiosk.Models;
using EdgeAIKiosk.Pipeline;

namespace EdgeAIKiosk.Services;

public sealed class ShoppingVerifier(
    ImageCapture imageCapture,
    IModelPreprocessor preprocessor,
    IModelLoader modelLoader,
    IObjectTrackingStrategy tracker) : System.IDisposable
{
    // Keeping capture, preprocessing, inference, and tracking separate lets the sample show
    // each edge-AI stage without hiding hardware/model assumptions in UI code.
    private readonly ImageCapture _imageCapture = imageCapture;
    private readonly IModelPreprocessor _preprocessor = preprocessor;
    private readonly IModelLoader _modelLoader = modelLoader;
    private readonly IObjectTrackingStrategy _tracker = tracker;

    // A short burst balances checkout latency with enough observations to smooth camera noise.
    private const int FrameCount = 5;

    /// <summary>Captures several frames, compares detected items against scanned items, and returns checkout status.</summary>
    /// <param name="scannedItems">The cart items collected from barcode scans before checkout.</param>
    public async Task<VerificationResult> Verify(List<ScannedItem> scannedItems)
    {
        _tracker.Track([], reset: true);
        for (int i = 0; i < FrameCount; i++)
        {
            SoftwareBitmap? frame = await _imageCapture.CaptureFrame();
            if (frame is null)
            {
                continue;
            }

            ModelInput input = _preprocessor.Preprocess(frame);
            ModelOutput output = await _modelLoader.RunInference(input);
            _tracker.Track(output.Detections);
        }
        return BuildVerificationResult(scannedItems, _tracker.Track([]));
    }

    /// <summary>Builds the final mismatch result from scanned cart items and tracked camera detections.</summary>
    /// <param name="scanned">The barcode cart items that the customer claims are present.</param>
    /// <param name="detected">The camera detections that passed the tracker threshold.</param>
    internal static VerificationResult BuildVerificationResult(
        List<ScannedItem> scanned,
        IReadOnlyList<DetectedItem> detected)
    {
        // The demo only compares labels that the selected model/classes can actually detect;
        // other scanned products need barcode/POS data, not camera verification.
        HashSet<string> scannedLabels = new(scanned.Select(item => item.Name).Where(KioskSettings.AcceptedLabels.Contains));
        HashSet<string> detectedLabels = new(detected.Select(item => item.Label).Where(KioskSettings.AcceptedLabels.Contains));

        // A mismatch is anything present in only one side.
        IEnumerable<string> scannedOnly = scannedLabels.Except(detectedLabels);
        IEnumerable<string> detectedOnly = detectedLabels.Except(scannedLabels);
        List<string> mismatches = scannedOnly.Concat(detectedOnly).Distinct().ToList();
        return new VerificationResult(mismatches.Count == 0, scanned, detected, mismatches);
    }

    public void Dispose()
    {
        _modelLoader.Dispose();
    }
}
