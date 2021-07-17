//
//  ImageViewRepresentable.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 9/7/21.
//

import Foundation
import UIKit
import SwiftUI


struct ImageView : UIViewRepresentable {
    @Binding var image : UIImage
    @Binding var predictions : [Prediction]
    
    func makeUIView(context: Context) -> UIImageView {
        let imageView = UIImageView(image: image)
        imageView.frame = CGRect(x: 0, y: 0, width: PrePostProcessor.inputWidth, height: PrePostProcessor.inputHeight)
        for prediction in predictions {
            let bbox = prediction.bbox
            let rect = CGRect(x: bbox.x1, y: bbox.y1, width: bbox.x2 - bbox.x1, height: bbox.y2 - bbox.y1)
            let bboxView = UIView(frame: rect)
            bboxView.backgroundColor = UIColor.clear
            bboxView.layer.borderColor = UIColor.green.cgColor
            bboxView.layer.borderWidth = 2
            imageView.addSubview(bboxView)
            
            let classIndex = prediction.classIndex
            let score = prediction.score
            let textLayer = CATextLayer()
            textLayer.string = String(format: "%@ %.2f", classIndex, score)
            textLayer.foregroundColor = UIColor.white.cgColor
            textLayer.backgroundColor = UIColor.magenta.cgColor
            textLayer.fontSize = 10
            textLayer.frame = CGRect(x: rect.origin.x, y: rect.origin.y, width: 20, height: 20)
            imageView.layer.addSublayer(textLayer)
        }
        return imageView
    }
    
    func updateUIView(_ uiView: UIImageView, context: Context) {
        
    }
}
