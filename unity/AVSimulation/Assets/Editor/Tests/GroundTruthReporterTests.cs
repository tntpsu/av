using NUnit.Framework;
using UnityEngine;

public class GroundTruthReporterTests
{
    [Test]
    public void GetLaneCenterPosition_OffsetsRightAndLeft()
    {
        Vector3 roadCenter = Vector3.zero;
        Vector3 roadDirection = Vector3.forward;
        float roadWidth = 7.2f;

        Vector3 rightLane = GroundTruthReporter.GetLaneCenterPosition(
            roadCenter, roadDirection, roadWidth, 1
        );
        Vector3 leftLane = GroundTruthReporter.GetLaneCenterPosition(
            roadCenter, roadDirection, roadWidth, 0
        );

        Assert.AreEqual(1.8f, rightLane.x, 1e-4f);
        Assert.AreEqual(-1.8f, leftLane.x, 1e-4f);
    }
}
