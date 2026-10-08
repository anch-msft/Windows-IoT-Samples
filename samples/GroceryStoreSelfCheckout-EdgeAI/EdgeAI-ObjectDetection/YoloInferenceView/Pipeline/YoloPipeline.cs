// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using EdgeAI_ObjectDetection.Controls;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;

namespace EdgeAI_ObjectDetection.Pipeline;

internal sealed class YoloPipeline : IAsyncDisposable
{
    private MediaCapture? _capture;
    private MediaFrameReader? _reader;
    private bool _readerStarted;
    private YoloInferenceEngine? _engine;
    private TimeSpan? _lastFrameTime;
    public event EventHandler<InferenceFaultedEventArgs>? Faulted;
    public MediaFrameSource? FrameSource
    {
        get;
        private set;
    }
    public string ProviderDescription => $"{_engine?.HardwareDevice}/{_engine?.ExecutionProvider}";

    public static bool IsColorVideoSource(MediaFrameSourceInfo info) =>
        info.SourceKind == MediaFrameSourceKind.Color &&
        (info.MediaStreamType == MediaStreamType.VideoPreview || info.MediaStreamType == MediaStreamType.VideoRecord);

    public async Task StartAsync(InferenceSettings settings, CancellationToken token)
    {
        _engine = await Task.Run(() => YoloInferenceEngine.Create(settings.ModelPath, settings.ExecutionProvider), token);
        token.ThrowIfCancellationRequested();
        _capture = new MediaCapture();
        _capture.Failed += Capture_Failed;
        await _capture.InitializeAsync(new MediaCaptureInitializationSettings
        {
            SourceGroup = settings.FrameSourceGroup,
            SharingMode = MediaCaptureSharingMode.SharedReadOnly,
            StreamingCaptureMode = StreamingCaptureMode.Video,
            MemoryPreference = MediaCaptureMemoryPreference.Cpu
        });
        token.ThrowIfCancellationRequested();
        FrameSource = _capture.FrameSources.Values
            .Where(source => IsColorVideoSource(source.Info))
            .OrderBy(source => source.Info.MediaStreamType == MediaStreamType.VideoPreview ? 0 : 1)
            .FirstOrDefault()
            ?? throw new InvalidOperationException("The camera group has no supported color video stream.");
        _reader = await _capture.CreateFrameReaderAsync(FrameSource, MediaEncodingSubtypes.Bgra8);
        token.ThrowIfCancellationRequested();
        MediaFrameReaderStartStatus status = await _reader.StartAsync();
        if (status != MediaFrameReaderStartStatus.Success)
        {
            throw new InvalidOperationException($"Could not start the camera reader: {status}.");
        }

        _readerStarted = true;
        token.ThrowIfCancellationRequested();
    }

    private void Capture_Failed(MediaCapture sender, MediaCaptureFailedEventArgs args) =>
        Faulted?.Invoke(this, new InferenceFaultedEventArgs(
            new InvalidOperationException($"Camera capture failed ({args.Code}): {args.Message}")));

    // Called serially on a worker thread. The caller awaits this work before disposal.
    public PipelineResult? TryInfer(PreviewGeometry geometry, int sourceWidth, int sourceHeight,
        bool captureImage, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        long startedAt = Stopwatch.GetTimestamp();
        using MediaFrameReference? frame = _reader!.TryAcquireLatestFrame();
        using SoftwareBitmap? bitmap = frame?.VideoMediaFrame?.SoftwareBitmap;
        if (bitmap is null || (frame!.SystemRelativeTime is { } time && time == _lastFrameTime))
        {
            return null;
        }

        _lastFrameTime = frame!.SystemRelativeTime;
        if (bitmap.PixelWidth != sourceWidth || bitmap.PixelHeight != sourceHeight)
        {
            return null; // A format change will be picked up by the next layout snapshot.
        }

        float[] input = YoloInferenceEngine.Preprocess(bitmap, geometry, captureImage, out byte[]? imagePixels);
        token.ThrowIfCancellationRequested();
        Stopwatch inference = Stopwatch.StartNew();
        IReadOnlyList<Detection> detections = _engine!.Run(input);
        inference.Stop();
        token.ThrowIfCancellationRequested();
        return new(detections, inference.Elapsed, startedAt, imagePixels);
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            if (_readerStarted)
            {
                await _reader!.StopAsync();
            }
        }
        finally
        {
            _readerStarted = false;
            try
            {
                _reader?.Dispose();
            }
            finally
            {
                _reader = null;
                FrameSource = null;
                try
                {
                    if (_capture is not null)
                    {
                        _capture.Failed -= Capture_Failed;
                        _capture.Dispose();
                    }
                }
                finally
                {
                    _capture = null;
                    _engine?.Dispose();
                    _engine = null;
                }
            }
        }
    }
}

internal sealed record PipelineResult(
    IReadOnlyList<Detection> Detections, TimeSpan InferenceTime, long StartedAt, byte[]? ImagePixels);
