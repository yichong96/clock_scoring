//
//  ObjectDetector.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 6/7/21.
//


class ObjectDetector {
    lazy var module: InferenceModule = {
        if let filePath = Bundle.main.path(forResource: "mobilenetv3_optimized", ofType: "pt"),
            let module = InferenceModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Failed to load model!")
        }
    }()
    
    lazy var classes: [String] = {
        if let filePath = Bundle.main.path(forResource: "classes", ofType: "txt"),
            let classes = try? String(contentsOfFile: filePath) {
            return classes.components(separatedBy: .newlines)
        } else {
            fatalError("classes file was not found.")
        }
    }()
}
