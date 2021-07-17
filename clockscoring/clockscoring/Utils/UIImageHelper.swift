//
//  UIImageHelper.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 6/7/21.
//

import Foundation
import UIKit
extension UIImage {
    func resized(to newSize: CGSize, scale: CGFloat = 1) -> UIImage {
       let format = UIGraphicsImageRendererFormat.default()
       format.scale = scale
       let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
       let image = renderer.image { _ in
           draw(in: CGRect(origin: .zero, size: newSize))
       }
       return image
    }
    
    func addBoundingBoxImage(predictions: [Prediction]) -> UIImage {
        var image = self
        for prediction in predictions {
            let box = prediction.bbox
            image = image.drawRectangleOnImage(box: box)
        }
        return image
    }
    
    func drawRectangleOnImage(box: BoundingBox) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: self.size)
        let image = renderer.image { (context) in
            draw(at: CGPoint.zero)
            let rectangle = CGRect(x: box.x1, y: box.y1, width: box.x2 - box.x1, height: box.y2 - box.y1)
            context.cgContext.setStrokeColor(UIColor.green.cgColor)
            context.cgContext.setLineWidth(CGFloat(2))
            context.cgContext.addRect(rectangle)
            context.cgContext.drawPath(using: .stroke)
        }
        return image
    }
    
    public convenience init?(color: UIColor, size: CGSize = CGSize(width: 1, height: 1)) {
        let rect = CGRect(origin: .zero, size: size)
        UIGraphicsBeginImageContextWithOptions(rect.size, false, 1.0)
        color.setFill()
        UIRectFill(rect)
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        guard let cgImage = image?.cgImage else { return nil }
        self.init(cgImage: cgImage)
    }
    
    
    func normalized() -> [Float32]? {
        guard let cgImage = self.cgImage else {
            return nil
        }
        let w = cgImage.width
        let h = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * w
        let bitsPerComponent = 8
        var rawBytes: [UInt8] = [UInt8](repeating:0, count: w * h * 4)
        // mutate rawBytes by rendering the bytes at the address ptr.baseAddress
        rawBytes.withUnsafeMutableBytes { ptr in
            if let cgImage = self.cgImage,
               let context = CGContext(data: ptr.baseAddress,
                                       width: w,
                                       height: h,
                                       bitsPerComponent: bitsPerComponent,
                                       bytesPerRow : bytesPerRow,
                                       space: CGColorSpaceCreateDeviceRGB(),
                                       bitmapInfo:  CGImageAlphaInfo.premultipliedLast.rawValue) {
                let rect = CGRect(x: 0, y: 0, width: w, height: h)
                context.draw(cgImage, in: rect)
            }
        }
        var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: w * h * 3)
        for i in 0 ..< w * h {
            normalizedBuffer[i] = Float32(rawBytes[i * 4 + 0]) / 255.0
            normalizedBuffer[w * h + i] = Float32(rawBytes[i * 4 + 1]) / 255.0
            normalizedBuffer[w * h * 2 + i] = Float32(rawBytes[i * 4 + 2]) / 255.0
        }
        return normalizedBuffer
    }
}
