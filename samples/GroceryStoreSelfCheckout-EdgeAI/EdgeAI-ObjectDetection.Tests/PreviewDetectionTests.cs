// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using EdgeAI_ObjectDetection.Controls;

namespace EdgeAI_ObjectDetection.Tests;

public sealed class PreviewDetectionTests
{
    private static readonly Detection ImageDetection =
        new(0, "person", .9f, new DetectionBox(.25f, .3125f, .5f, .375f));

    [Fact]
    public void LivePreviewRemovesModelPadding()
    {
        var geometry = PreviewGeometry.Create(640, 480, 640, 480, 1);
        var detection = Assert.Single(geometry.MapDetections([ImageDetection], LetterboxGeometry.Create(640, 480)));

        Assert.Equal(new PreviewDetectionBox(160, 120, 320, 240), detection.BoundingBox);
        Assert.Equal(ImageDetection.ClassId, detection.ClassId);
        Assert.Equal(ImageDetection.Label, detection.Label);
        Assert.Equal(ImageDetection.Confidence, detection.Confidence);
    }

    [Fact]
    public void LivePreviewIncludesDisplayOffsetsAndScale()
    {
        var geometry = PreviewGeometry.Create(640, 480, 1280, 1280, 2);
        var detection = Assert.Single(geometry.MapDetections([ImageDetection], LetterboxGeometry.Create(640, 480)));

        Assert.Equal(new PreviewDetectionBox(320, 400, 640, 480), detection.BoundingBox);
    }

    [Fact]
    public void CroppedPreviewMapsToVisibleRegion()
    {
        var geometry = PreviewGeometry.Create(1920, 1080, 400, 800, 800.0 / 1080);
        Detection detection = new(1, "bicycle", .8f, new DetectionBox(.375f, .25f, .25f, .5f));
        var padding = LetterboxGeometry.Create(geometry.CropWidth, geometry.CropHeight);
        var box = Assert.Single(geometry.MapDetections([detection], padding)).BoundingBox;

        Assert.Equal(100, box.X, 5);
        Assert.Equal(200, box.Y, 5);
        Assert.Equal(200, box.Width, 5);
        Assert.Equal(400, box.Height, 5);
    }

    [Fact]
    public void SnapshotDisplayPreservesImagePaddingAndImageCoordinates()
    {
        var geometry = PreviewGeometry.FitSquare(640, 800, 600);
        var detection = Assert.Single(geometry.MapDetections([ImageDetection], null));

        Assert.Equal(new PreviewDetectionBox(250, 187.5, 300, 225), detection.BoundingBox);
        Assert.Equal(new DetectionBox(.25f, .3125f, .5f, .375f), ImageDetection.BoundingBox);
    }

    [Fact]
    public void ResizingProducesNewCoordinatesWithoutMutatingPreviousResults()
    {
        var first = PreviewGeometry.FitSquare(640, 640, 640).MapDetections([ImageDetection], null);
        var resized = PreviewGeometry.FitSquare(640, 1280, 1280).MapDetections([ImageDetection], null);

        Assert.Equal(new PreviewDetectionBox(160, 200, 320, 240), first[0].BoundingBox);
        Assert.Equal(new PreviewDetectionBox(320, 400, 640, 480), resized[0].BoundingBox);
        Assert.Throws<NotSupportedException>(() => ((IList<PreviewDetection>)first).Clear());
        Assert.Empty(PreviewGeometry.FitSquare(640, 640, 640).MapDetections([], null));
    }
}
