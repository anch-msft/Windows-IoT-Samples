using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using EdgeAI_ObjectDetection.Controls;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Graphics.Imaging;

namespace EdgeAI_ObjectDetection.Pipeline;

internal sealed class YoloInferenceEngine : IDisposable
{
    internal const int InputSize = 640;
    internal const int DetectionLimit = 300;

    private static readonly string[] CocoLabels =
    [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ];

    private readonly InferenceSession _session;
    public static IReadOnlyList<string> SupportedClasses
    {
        get;
    } = Array.AsReadOnly(CocoLabels);

    public string ExecutionProvider
    {
        get;
    }
    public string HardwareDevice
    {
        get;
    }

    private YoloInferenceEngine(InferenceSession session, string executionProvider, string hardwareDevice)
    {
        _session = session;
        ExecutionProvider = executionProvider;
        HardwareDevice = hardwareDevice;
    }

    public static YoloInferenceEngine Create(string modelPath, OrtEpDevice selectedDevice)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);
        }

        InferenceSession session;
        using (SessionOptions options = new())
        {
            options.AppendExecutionProvider(OrtEnv.Instance(), [selectedDevice], new Dictionary<string, string>());
            session = new InferenceSession(modelPath, options);
        }
        try
        {
            if (session.InputMetadata.Count != 1 || session.OutputMetadata.Count != 1)
            {
                throw new InvalidOperationException("Expected exactly one YOLO input and one output.");
            }

            NodeMetadata input = session.InputMetadata.Values.Single();
            NodeMetadata output = session.OutputMetadata.Values.Single();
            if (!input.IsTensor || input.ElementType != typeof(float) ||
                !output.IsTensor || output.ElementType != typeof(float))
            {
                throw new InvalidOperationException("Expected float32 input and output tensors.");
            }

            ValidateShapes(input.Dimensions, output.Dimensions);
            return new YoloInferenceEngine(session, selectedDevice.EpName, selectedDevice.HardwareDevice.Type.ToString());
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    internal static void ValidateShapes(IEnumerable<int> input, IEnumerable<int> output)
    {
        if (!input.SequenceEqual(new[] { 1, 3, InputSize, InputSize }) ||
            !output.SequenceEqual(new[] { 1, DetectionLimit, 6 }))
        {
            throw new InvalidOperationException(
                "Supported model contract: float32 input [1,3,640,640] and output [1,300,6], processed COCO detections.");
        }
    }

    internal static float[] Preprocess(SoftwareBitmap frame, PreviewGeometry crop,
        bool captureImage, out byte[]? imagePixels)
    {
        using SoftwareBitmap rgbaFrame = SoftwareBitmap.Convert(
            frame,
            BitmapPixelFormat.Rgba8,
            BitmapAlphaMode.Ignore);
        int sourceWidth = rgbaFrame.PixelWidth;
        int sourceHeight = rgbaFrame.PixelHeight;
        byte[] rgbaPixels = new byte[sourceWidth * sourceHeight * 4];
        rgbaFrame.CopyToBuffer(rgbaPixels.AsBuffer());

        using Image<Rgba32> image = Image.LoadPixelData<Rgba32>(rgbaPixels, sourceWidth, sourceHeight);
        image.Mutate(operation => operation
            .Crop(new Rectangle(crop.CropX, crop.CropY, crop.CropSize, crop.CropSize))
            .Resize(InputSize, InputSize));

        float[] tensor = new float[3 * InputSize * InputSize];
        byte[]? snapshotPixels = captureImage ? new byte[4 * InputSize * InputSize] : null;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < InputSize; y++)
            {
                Span<Rgba32> row = accessor.GetRowSpan(y);
                for (int x = 0; x < InputSize; x++)
                {
                    int pixelIndex = y * InputSize + x;
                    tensor[pixelIndex] = row[x].R / 255f;
                    tensor[InputSize * InputSize + pixelIndex] = row[x].G / 255f;
                    tensor[2 * InputSize * InputSize + pixelIndex] = row[x].B / 255f;
                    if (snapshotPixels is not null)
                    {
                        // Copy the same resized pixels used by the tensor, not a later camera frame.
                        int offset = pixelIndex * 4;
                        snapshotPixels[offset] = row[x].B;
                        snapshotPixels[offset + 1] = row[x].G;
                        snapshotPixels[offset + 2] = row[x].R;
                        snapshotPixels[offset + 3] = 255;
                    }
                }
            }
        });

        imagePixels = snapshotPixels;
        return tensor;
    }

    public IReadOnlyList<Detection> Run(float[] input)
    {
        string inputName = _session.InputMetadata.Keys.Single();
        DenseTensor<float> tensor = new(input, [1, 3, InputSize, InputSize]);
        NamedOnnxValue modelInput = NamedOnnxValue.CreateFromTensor(inputName, tensor);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = _session.Run([modelInput]);
        DisposableNamedOnnxValue output = outputs.FirstOrDefault()
            ?? throw new InvalidOperationException("The selected model returned no output tensors.");
        return ParseDetections(output.AsTensor<float>());
    }

    private static IReadOnlyList<Detection> ParseDetections(Tensor<float> output)
    {
        if (output.Rank != 3 || output.Dimensions[0] != 1 ||
            output.Dimensions[1] != DetectionLimit || output.Dimensions[2] != 6)
        {
            throw new InvalidOperationException(
                "Expected processed YOLO output [1,300,6] (x1, y1, x2, y2, confidence, COCO class id).");
        }

        List<Detection> detections = new();
        for (int index = 0; index < output.Dimensions[1]; index++)
        {
            Detection? detection = DecodeDetection(
                output[0, index, 0], output[0, index, 1], output[0, index, 2], output[0, index, 3],
                output[0, index, 4], output[0, index, 5], SupportedClasses);
            if (detection is not null)
            {
                detections.Add(detection);
            }
        }

        return detections.AsReadOnly();
    }

    internal static Detection? DecodeDetection(float x1, float y1, float x2, float y2,
        float confidence, float classId, IReadOnlyList<string> labels)
    {
        if (!float.IsFinite(confidence) || confidence < 0.5f || confidence > 1 ||
            !float.IsFinite(classId) || classId < 0 || classId >= labels.Count || classId != (int)classId ||
            !float.IsFinite(x1) || !float.IsFinite(y1) || !float.IsFinite(x2) || !float.IsFinite(y2))
        {
            return null;
        }

        x1 = Math.Clamp(x1 / InputSize, 0, 1);
        y1 = Math.Clamp(y1 / InputSize, 0, 1);
        x2 = Math.Clamp(x2 / InputSize, 0, 1);
        y2 = Math.Clamp(y2 / InputSize, 0, 1);
        if (x2 <= x1 || y2 <= y1)
        {
            return null;
        }
        return new Detection((int)classId, labels[(int)classId], confidence,
            new DetectionBox(x1, y1, x2 - x1, y2 - y1));
    }

    public void Dispose() => _session.Dispose();
}
