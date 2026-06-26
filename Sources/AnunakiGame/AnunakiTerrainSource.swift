import Foundation

public enum AnunakiTerrainSource: String, Sendable, Hashable {
    case trainedTerrainDiffuser = "TRAINED DIFFUSER"
    case debugDiffuser = "DEBUG DIFFUSER"
    case modelArchivesMissing = "MODEL ARCHIVES MISSING"
    case proceduralFallback = "FALLBACK"
}

public struct AnunakiTerrainBootstrap: Sendable {
    public let provider: AnunakiTerrainProvider
    public let status: AnunakiTerrainSource
    public let archiveRoot: URL?
}
