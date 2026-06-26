// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "MetalTerrainDiffusionPro",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MetalTerrainDiffusion", targets: ["MetalTerrainDiffusion"]),
        .library(name: "AnunakiGame", targets: ["AnunakiGame"]),
        .executable(name: "terrain-diffusion-metal", targets: ["TerrainDiffusionCLI"])
    ],
    targets: [
        .target(
            name: "MetalTerrainDiffusion",
            resources: [.process("Shaders")],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Accelerate")
            ]
        ),
        .target(
            name: "AnunakiGame",
            dependencies: ["MetalTerrainDiffusion"],
            linkerSettings: [
                .linkedFramework("SceneKit"),
                .linkedFramework("SwiftUI")
            ]
        ),
        .executableTarget(name: "TerrainDiffusionCLI", dependencies: ["MetalTerrainDiffusion"]),
        .testTarget(name: "MetalTerrainDiffusionTests", dependencies: ["MetalTerrainDiffusion"]),
        .testTarget(name: "AnunakiGameTests", dependencies: ["AnunakiGame"])
    ]
)
