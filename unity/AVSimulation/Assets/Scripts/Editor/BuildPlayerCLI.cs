using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildPlayerCLI
{
    public static void BuildMacPlayer()
    {
        string outputPath = GetCommandLineArg("-buildOutput");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            outputPath = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "mybuild.app")
            );
        }
        else
        {
            outputPath = Path.GetFullPath(outputPath);
        }

        string[] scenes = EditorBuildSettings.scenes
            .Where(scene => scene.enabled)
            .Select(scene => scene.path)
            .ToArray();

        if (scenes.Length == 0)
        {
            Debug.LogError("BuildPlayerCLI: No enabled scenes in Build Settings.");
            EditorApplication.Exit(1);
            return;
        }

        Debug.Log($"BuildPlayerCLI: Building macOS player to {outputPath}");

        BuildPlayerOptions buildOptions = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = outputPath,
            target = BuildTarget.StandaloneOSX,
            options = BuildOptions.None
        };

        BuildReport report = BuildPipeline.BuildPlayer(buildOptions);
        BuildSummary summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"BuildPlayerCLI: Build succeeded ({summary.totalSize} bytes)");
            EditorApplication.Exit(0);
        }
        else
        {
            Debug.LogError($"BuildPlayerCLI: Build failed ({summary.result})");
            EditorApplication.Exit(1);
        }
    }

    private static string GetCommandLineArg(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == name && i + 1 < args.Length)
            {
                return args[i + 1];
            }
        }

        return null;
    }
}
