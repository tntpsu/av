using UnityEngine;

/// <summary>
/// Ensures the Unity player continues running when the app is not focused.
/// </summary>
public static class RunInBackground
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void EnableRunInBackground()
    {
        Application.runInBackground = true;
        QualitySettings.vSyncCount = 0;
        if (Application.targetFrameRate <= 0)
        {
            Application.targetFrameRate = 60;
        }

        Debug.Log($"RunInBackground: runInBackground={Application.runInBackground}, " +
                  $"targetFrameRate={Application.targetFrameRate}, vSyncCount={QualitySettings.vSyncCount}");
    }
}
