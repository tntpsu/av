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
        // Shader.WarmupAllShaders() deterministically crashes in NEON matrix
        // code on Apple Silicon under Unity 6 + macOS 26 (EXC_BAD_ACCESS at
        // 0x4e4f5f, reproduced 2026-04-16). Unity 6's default async shader
        // compilation replaces the need for explicit prewarmup — shaders
        // compile in the background with a placeholder substitute, avoiding
        // both the upfront cost AND the per-frame stalls this prewarmer was
        // originally designed to prevent.
        Debug.Log("ShaderPrewarmer: [BeforeSceneLoad] skipped — Unity 6 async compilation handles this.");
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
            // Disabled for same reason as PrewarmShadersEarly() — crashes on
            // Apple Silicon + Unity 6. Async compilation is the replacement.
            Debug.Log("ShaderPrewarmer: [Awake] generic warmup skipped — Unity 6 async compilation handles this.");
        }
    }
}
