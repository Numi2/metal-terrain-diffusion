import Foundation
import simd

public struct AnunakiFlightInput: Sendable, Equatable {
    public var throttle: Float
    public var yaw: Float
    public var pitch: Float
    public var boost: Bool
    public var tractorBeam: Bool

    public init(throttle: Float = 0, yaw: Float = 0, pitch: Float = 0, boost: Bool = false, tractorBeam: Bool = false) {
        self.throttle = throttle.clamped(to: 0...1)
        self.yaw = yaw.clamped(to: -1...1)
        self.pitch = pitch.clamped(to: -1...1)
        self.boost = boost
        self.tractorBeam = tractorBeam
    }
}

public struct AnunakiShipState: Sendable, Equatable {
    public var position: SIMD3<Float>
    public var velocity: SIMD3<Float>
    public var yaw: Float
    public var pitch: Float
    public var energy: Float
    public var integrity: Float
    public var score: Int

    public init(
        position: SIMD3<Float> = SIMD3<Float>(0, 160, 0),
        velocity: SIMD3<Float> = SIMD3<Float>(0, 0, -45),
        yaw: Float = 0,
        pitch: Float = -0.05,
        energy: Float = 1,
        integrity: Float = 1,
        score: Int = 0
    ) {
        self.position = position
        self.velocity = velocity
        self.yaw = yaw
        self.pitch = pitch
        self.energy = energy.clamped(to: 0...1)
        self.integrity = integrity.clamped(to: 0...1)
        self.score = score
    }

    public var speed: Float { simd_length(velocity) }
}

public struct AnunakiFlightModel: Sendable {
    public var state: AnunakiShipState
    public var minimumClearance: Float
    public var gravity: Float
    public var lift: Float
    public var drag: Float
    public var turnRate: Float
    public var pitchRate: Float

    public init(
        state: AnunakiShipState = AnunakiShipState(),
        minimumClearance: Float = 16,
        gravity: Float = 16,
        lift: Float = 26,
        drag: Float = 0.055,
        turnRate: Float = 1.9,
        pitchRate: Float = 1.15
    ) {
        self.state = state
        self.minimumClearance = minimumClearance
        self.gravity = gravity
        self.lift = lift
        self.drag = drag
        self.turnRate = turnRate
        self.pitchRate = pitchRate
    }

    public mutating func step(input: AnunakiFlightInput, deltaTime rawDeltaTime: Float, terrainHeight: Float) {
        let deltaTime = rawDeltaTime.clamped(to: 0...0.05)
        guard deltaTime > 0 else { return }

        state.yaw += input.yaw * turnRate * deltaTime
        state.pitch = (state.pitch + input.pitch * pitchRate * deltaTime).clamped(to: -0.72...0.52)

        let forward = SIMD3<Float>(sin(state.yaw) * cos(state.pitch), -sin(state.pitch), -cos(state.yaw) * cos(state.pitch))
        let boostAllowed = input.boost && state.energy > 0.08
        let thrust: Float = boostAllowed ? 148 : 82
        let hoverBias = max(0, 1 - abs((state.position.y - terrainHeight - 82) / 120))
        let upForce = lift * (0.45 + input.throttle * 0.9 + hoverBias * 0.3)

        state.velocity += forward * (thrust * input.throttle * deltaTime)
        state.velocity.y += (upForce - gravity) * deltaTime
        state.velocity -= state.velocity * drag * deltaTime

        if boostAllowed {
            state.velocity += forward * (62 * deltaTime)
            state.energy = max(0, state.energy - 0.18 * deltaTime)
        } else {
            state.energy = min(1, state.energy + 0.065 * deltaTime + (input.tractorBeam ? 0.02 * deltaTime : 0))
        }

        state.position += state.velocity * deltaTime

        let floor = terrainHeight + minimumClearance
        if state.position.y < floor {
            let impact = max(0, -state.velocity.y) / 80
            state.integrity = max(0, state.integrity - impact * 0.08)
            state.position.y = floor
            state.velocity.y = abs(state.velocity.y) * 0.32 + 8
            state.velocity.x *= 0.72
            state.velocity.z *= 0.72
        }
    }
}

extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
