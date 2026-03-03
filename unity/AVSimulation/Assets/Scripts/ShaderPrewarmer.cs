using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Pre-compiles all shader variants at startup to eliminate frame hitching
/// caused by on-demand shader compilation during the first few camera frames.
///
/// Runs before scene load via RuntimeInitializeOnLoadMethod, ensuring shaders
/// are compiled before CameraCapture.Start() fires.
///
/// Optional: Assign a ShaderVariantCollection asset for targeted warmup of
/// known variants. Falls back to Shader.WarmupAllShaders() which covers
/// all currently-loaded shaders.
/// </summary>
public class ShaderPrewarmer : MonoBehaviour
{
    [Tooltip("Optional ShaderVariantCollection for targeted prewarming. " +
             "If null, uses Shader.WarmupAllShaders() as fallback.")]
    public ShaderVariantCollection variantCollection;

    [Tooltip("Call Shader.WarmupAllShaders() in addition to (or instead of) the collection.")]
    public bool useGenericWarmup = true;

    /// <summary>
    /// Static initializer runs before any scene MonoBehaviour.Awake().
    /// This is the earliest point where we can warm shaders without
    /// requiring a MonoBehaviour to be attached to a GameObject.
    /// </summary>
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void PrewarmShadersEarly()
    {
        // WarmupAllShaders forces compilation of every loaded shader variant.
        // This may take 100-500ms on first launch but eliminates per-frame
        // compilation stalls that cause 200-300ms gaps in camera delivery.
        Shader.WarmupAllShaders();
        Debug.Log("ShaderPrewarmer: [BeforeSceneLoad] WarmupAllShaders complete.");
    }

    /// <summary>
    /// Instance-level warmup for ShaderVariantCollection assets that need
    /// to be assigned via the Inspector. Runs in Awake() (before Start()).
    /// </summary>
    void Awake()
    {
        if (variantCollection != null)
        {
            variantCollection.WarmUp();
            Debug.Log($"ShaderPrewarmer: Warmed ShaderVariantCollection " +
                      $"({variantCollection.shaderCount} shaders, " +
                      $"{variantCollection.variantCount} variants).");
        }

        if (useGenericWarmup)
        {
            // Second pass catches any shaders loaded between BeforeSceneLoad
            // and scene Awake (e.g., shaders on dynamically-spawned prefabs).
            Shader.WarmupAllShaders();
            Debug.Log("ShaderPrewarmer: [Awake] WarmupAllShaders complete.");
        }
    }
}
