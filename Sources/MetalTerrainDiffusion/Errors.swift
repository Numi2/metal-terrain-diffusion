import Foundation

public enum TerrainDiffusionError: Error, CustomStringConvertible {
    case metalUnavailable
    case shaderLoadFailed(String)
    case pipelineCreationFailed(String)
    case allocationFailed(Int)
    case commandBufferFailed(String)
    case invalidShape(String)
    case unsupportedScalarType(String)
    case missingTensor(String)
    case invalidGraph(String)
    case invalidModelArchive(String)
    case io(String)

    public var description: String {
        switch self {
        case .metalUnavailable: return "No Apple Metal device is available."
        case .shaderLoadFailed(let s): return "Failed to load Metal shader library: \(s)"
        case .pipelineCreationFailed(let s): return "Failed to create Metal pipeline: \(s)"
        case .allocationFailed(let n): return "Failed to allocate \(n) bytes."
        case .commandBufferFailed(let s): return "Metal command buffer failed: \(s)"
        case .invalidShape(let s): return "Invalid tensor shape: \(s)"
        case .unsupportedScalarType(let s): return "Unsupported scalar type: \(s)"
        case .missingTensor(let s): return "Missing tensor: \(s)"
        case .invalidGraph(let s): return "Invalid graph: \(s)"
        case .invalidModelArchive(let s): return "Invalid model archive: \(s)"
        case .io(let s): return "I/O error: \(s)"
        }
    }
}
