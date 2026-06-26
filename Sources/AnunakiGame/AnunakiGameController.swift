#if os(iOS)
import CoreGraphics
import SwiftUI

@MainActor
public final class AnunakiGameController: ObservableObject {
    @Published public private(set) var state = AnunakiShipState()
    @Published public private(set) var missionText = "ANUNAKI"
    @Published public private(set) var ringsCollected = 0
    @Published public private(set) var terrainSource = AnunakiTerrainSource.modelArchivesMissing
    @Published public private(set) var modelStatus = AnunakiTerrainSource.modelArchivesMissing

    public let seed: UInt64
    var input = AnunakiFlightInput(throttle: 0.72)

    public init(seed: UInt64) {
        self.seed = seed
    }

    func setStick(_ vector: CGVector) {
        input.yaw = Float(vector.dx).clamped(to: -1...1)
        input.pitch = Float(vector.dy).clamped(to: -1...1)
    }

    func setThrottle(_ value: Float) {
        input.throttle = value.clamped(to: 0...1)
    }

    func setBoost(_ active: Bool) {
        input.boost = active
    }

    func pulseTractorBeam() {
        input.tractorBeam = true
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 200_000_000)
            input.tractorBeam = false
        }
    }

    func publishFrame(state: AnunakiShipState, ringsCollected: Int) {
        self.state = state
        self.ringsCollected = ringsCollected
        missionText = Self.missionText(for: state, ringsCollected: ringsCollected)
    }

    func publishTerrainSource(_ terrainSource: AnunakiTerrainSource) {
        self.terrainSource = terrainSource
    }

    func publishModelStatus(_ modelStatus: AnunakiTerrainSource) {
        self.modelStatus = modelStatus
    }

    private static func missionText(for state: AnunakiShipState, ringsCollected: Int) -> String {
        if state.integrity <= 0.05 { return "HULL CRITICAL" }
        if state.energy < 0.18 { return "ENERGY LOW" }
        if ringsCollected > 0 && ringsCollected % 5 == 0 { return "ARTIFACT STREAM \(ringsCollected)" }
        return "ANUNAKI"
    }
}
#endif
