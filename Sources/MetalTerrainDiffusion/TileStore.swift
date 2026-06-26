import Foundation
import Metal

public protocol TileStore: AnyObject {
    func contains(_ key: WindowKey) -> Bool
    func get(_ key: WindowKey) -> MetalTensor?
    func put(_ tensor: MetalTensor, for key: WindowKey, byteCost: Int)
    func removeAll(prefix tensorID: String?)
    var residentByteCount: Int { get }
}

private final class LRUNode { let key: WindowKey; var tensor: MetalTensor; var cost: Int; var prev: LRUNode?; var next: LRUNode?; init(_ key: WindowKey,_ tensor: MetalTensor,_ cost: Int){self.key=key;self.tensor=tensor;self.cost=cost} }

public final class LRUMetalTileStore: TileStore {
    private let limitBytes: Int
    private var map: [WindowKey: LRUNode] = [:]
    private var head: LRUNode?
    private var tail: LRUNode?
    private var bytes = 0
    private let lock = NSLock()
    public init(limitBytes: Int = 1024 * 1024 * 1024) { self.limitBytes = max(limitBytes, 1 << 20) }
    public var residentByteCount: Int { lock.lock(); defer{lock.unlock()}; return bytes }
    public func contains(_ key: WindowKey) -> Bool { lock.lock(); defer{lock.unlock()}; return map[key] != nil }
    public func get(_ key: WindowKey) -> MetalTensor? { lock.lock(); defer{lock.unlock()}; guard let n=map[key] else { return nil }; touch(n); return n.tensor }
    public func put(_ tensor: MetalTensor, for key: WindowKey, byteCost: Int) { lock.lock(); defer{lock.unlock()}; if let n=map[key]{bytes-=n.cost;n.tensor=tensor;n.cost=byteCost;bytes+=byteCost;touch(n)} else { let n=LRUNode(key,tensor,byteCost); map[key]=n; insert(n); bytes+=byteCost }; evict() }
    public func removeAll(prefix tensorID: String? = nil) { lock.lock(); defer{lock.unlock()}; if let p=tensorID { for k in map.keys where k.tensorID == p { if let n=map.removeValue(forKey:k){ detach(n); bytes-=n.cost } } } else { map.removeAll(); head=nil; tail=nil; bytes=0 } }
    private func insert(_ n: LRUNode){ n.next=head; head?.prev=n; head=n; if tail == nil { tail=n } }
    private func detach(_ n: LRUNode){ let p=n.prev, q=n.next; p?.next=q; q?.prev=p; if head === n { head=q }; if tail === n { tail=p }; n.prev=nil; n.next=nil }
    private func touch(_ n: LRUNode){ if head !== n { detach(n); insert(n) } }
    private func evict(){ while bytes > limitBytes, let t=tail { map.removeValue(forKey:t.key); detach(t); bytes-=t.cost } }
}

public final class PersistentTileStore: TileStore {
    public struct Header: Codable { var tensorID:String; var y:Int; var x:Int; var n:Int; var c:Int; var h:Int; var w:Int; var scalarType:ScalarType; var byteCount:Int }
    private let context: MetalContext
    private let root: URL
    private let memory: LRUMetalTileStore
    private let fm = FileManager.default
    public init(context: MetalContext, root: URL, memoryLimitBytes: Int = 512 * 1024 * 1024) throws { self.context=context; self.root=root; self.memory=LRUMetalTileStore(limitBytes: memoryLimitBytes); try fm.createDirectory(at: root, withIntermediateDirectories: true) }
    public var residentByteCount: Int { memory.residentByteCount }
    public func contains(_ key: WindowKey) -> Bool { memory.contains(key) || fm.fileExists(atPath: path(key).path) }
    public func get(_ key: WindowKey) -> MetalTensor? { if let t=memory.get(key){return t}; guard let data=try? Data(contentsOf:path(key)), data.count>4 else { return nil }; let hsz=Int(data.withUnsafeBytes{$0.load(as:UInt32.self)}); guard data.count >= 4+hsz else { return nil }; guard let h=try? JSONDecoder().decode(Header.self, from:data.subdata(in:4..<(4+hsz))) else { return nil }; let payload=data.subdata(in:(4+hsz)..<data.count); do { let shape=try TensorShape(n:h.n,c:h.c,h:h.h,w:h.w); guard let staging=context.device.makeBuffer(length:payload.count, options:.storageModeShared) else { return nil }; payload.copyBytes(to: staging.contents().bindMemory(to: UInt8.self, capacity: payload.count), count: payload.count); let out=try context.allocate(shape:shape, scalarType:h.scalarType, storageMode:.storageModePrivate, label:key.description); let cb=try context.makeCommandBuffer(label:"tile.load"); let blit=cb.makeBlitCommandEncoder()!; blit.copy(from:staging, sourceOffset:0, to:out.buffer, destinationOffset:0, size:payload.count); blit.endEncoding(); try context.runAndWait(cb); memory.put(out, for:key, byteCost:out.byteCount); return out } catch { return nil } }
    public func put(_ tensor: MetalTensor, for key: WindowKey, byteCost: Int) { memory.put(tensor, for:key, byteCost:byteCost); guard tensor.scalarType == .float32 else { return }; do { let floats=try context.download(tensor); let h=Header(tensorID:key.tensorID,y:key.y,x:key.x,n:tensor.shape.n,c:tensor.shape.c,h:tensor.shape.h,w:tensor.shape.w,scalarType:tensor.scalarType,byteCount:tensor.byteCount); let hd=try JSONEncoder().encode(h); var out=Data(); var hsz=UInt32(hd.count); out.append(Data(bytes:&hsz,count:4)); out.append(hd); out.append(floats.withUnsafeBufferPointer{Data(buffer:$0)}); try fm.createDirectory(at:path(key).deletingLastPathComponent(), withIntermediateDirectories:true); try out.write(to:path(key), options:.atomic) } catch {} }
    public func removeAll(prefix tensorID: String? = nil) { memory.removeAll(prefix:tensorID); if let t=tensorID { try? fm.removeItem(at:root.appendingPathComponent(safe(t), isDirectory:true)) } else { try? fm.removeItem(at:root); try? fm.createDirectory(at:root, withIntermediateDirectories:true) } }
    private func safe(_ s:String)->String{s.replacingOccurrences(of:"/",with:"_").replacingOccurrences(of:" ",with:"_")}
    private func path(_ key: WindowKey)->URL{ root.appendingPathComponent(safe(key.tensorID), isDirectory:true).appendingPathComponent("\(key.y)_\(key.x).tdtile") }
}
