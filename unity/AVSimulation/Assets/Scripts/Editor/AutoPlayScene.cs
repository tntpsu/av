using UnityEngine;
using UnityEditor;

/// <summary>
/// Editor script that automatically enters play mode when Unity opens the project.
/// This is useful for automated testing and headless execution.
/// </summary>
[InitializeOnLoad]
public class AutoPlayScene : EditorWindow
{
    private static bool autoPlayEnabled = false;
    private static bool hasAutoPlayed = false;
    
    static AutoPlayScene()
    {
        // Check for auto-play flag file (more reliable than environment variables)
        string flagFile = System.IO.Path.Combine(Application.dataPath, "..", ".unity_autoplay");
        if (System.IO.File.Exists(flagFile))
        {
            autoPlayEnabled = true;
            // Delete flag file after reading
            try { System.IO.File.Delete(flagFile); } catch { }
        }
        
        // Check if auto-play is enabled via command line argument
        string[] args = System.Environment.GetCommandLineArgs();
        foreach (string arg in args)
        {
            if (arg == "-autoPlay" || arg == "-executeMethod")
            {
                autoPlayEnabled = true;
                break;
            }
        }
        
        // Also check for environment variable (may not work from command line)
        string envAutoPlay = System.Environment.GetEnvironmentVariable("UNITY_AUTO_PLAY");
        if (envAutoPlay == "1" || envAutoPlay == "true")
        {
            autoPlayEnabled = true;
        }
        
        // Register callback for when play mode state changes
        EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        
        // Register callback for when editor is quitting
        EditorApplication.quitting += OnEditorQuitting;
        
        // Auto-enter play mode if enabled and not already playing
        if (autoPlayEnabled && !EditorApplication.isPlaying && !hasAutoPlayed)
        {
            Debug.Log("AutoPlayScene: Auto-play enabled, entering play mode...");
            EditorApplication.delayCall += EnterPlayMode;
        }
    }
    
    private static void EnterPlayMode()
    {
        if (!EditorApplication.isPlaying && !hasAutoPlayed)
        {
            Debug.Log("AutoPlayScene: Entering play mode automatically");
            hasAutoPlayed = true;
            
            // Wait a moment for Unity to fully initialize before entering play mode
            EditorApplication.delayCall += () => {
                if (!EditorApplication.isPlaying)
                {
                    Debug.Log("AutoPlayScene: Actually entering play mode now...");
                    EditorApplication.isPlaying = true;
                }
            };
        }
    }
    
    private static void OnPlayModeStateChanged(PlayModeStateChange state)
    {
        if (state == PlayModeStateChange.ExitingPlayMode)
        {
            hasAutoPlayed = false;
        }
    }
    
    private static void OnEditorQuitting()
    {
        // Exit play mode before Unity quits (graceful shutdown)
        if (EditorApplication.isPlaying)
        {
            Debug.Log("AutoPlayScene: Editor quitting, exiting play mode gracefully...");
            EditorApplication.isPlaying = false;
            // Note: Unity will wait for play mode to exit before quitting
        }
    }
    
    [MenuItem("AV Stack/Auto Play Settings")]
    public static void ShowWindow()
    {
        GetWindow<AutoPlayScene>("Auto Play Settings");
    }
    
    void OnGUI()
    {
        GUILayout.Label("Auto Play Configuration", EditorStyles.boldLabel);
        GUILayout.Space(10);
        
        EditorGUILayout.HelpBox(
            "Auto-play can be enabled by:\n" +
            "1. Creating file: .unity_autoplay in project root (automated)\n" +
            "2. Setting environment variable: UNITY_AUTO_PLAY=1\n" +
            "3. Using command line: -autoPlay\n" +
            "4. Or manually enabling below",
            MessageType.Info
        );
        
        GUILayout.Space(10);
        
        autoPlayEnabled = EditorGUILayout.Toggle("Enable Auto Play", autoPlayEnabled);
        
        if (GUILayout.Button("Enter Play Mode Now"))
        {
            if (!EditorApplication.isPlaying)
            {
                EditorApplication.isPlaying = true;
            }
        }
        
        if (GUILayout.Button("Exit Play Mode"))
        {
            if (EditorApplication.isPlaying)
            {
                EditorApplication.isPlaying = false;
            }
        }
    }
}

