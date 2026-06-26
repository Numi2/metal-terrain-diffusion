import Foundation

public struct EDMSchedulerConfig: Codable, Sendable, Hashable { public var sigmaMin:Float=0.002; public var sigmaMax:Float=80; public var sigmaData:Float=0.5; public var rho:Float=7; public var finalSigma:Float=0; public init(){} }
public struct EDMScheduler: Sendable {
    public var config: EDMSchedulerConfig
    public init(config: EDMSchedulerConfig = EDMSchedulerConfig()){self.config=config}
    public func karrasSigmas(steps: Int)->[Float]{ guard steps>0 else{return[]}; if steps==1{return[config.sigmaMax]}; let minInv=pow(config.sigmaMin,1/config.rho), maxInv=pow(config.sigmaMax,1/config.rho); return (0..<steps).map{let r=Float($0)/Float(steps-1); return pow(maxInv + r*(minInv-maxInv), config.rho)} }
    public func preconditionNoise(sigma: Float)->Float { 0.25 * log(max(sigma,1e-20)) }
    public func trigflowTime(sigma: Float)->Float { atan(sigma / config.sigmaData) }
    public static func paperLatentIntermediate()->Float { atan(0.35 / 0.5) }
}
public struct DiffusionPhaseParameters: Codable, Sendable, Hashable { public var time: Float; public var seedOffset: UInt64; public var sigmaData: Float; public var name: String; public init(time:Float, seedOffset:UInt64, sigmaData:Float=0.5, name:String){self.time=time;self.seedOffset=seedOffset;self.sigmaData=sigmaData;self.name=name} }
