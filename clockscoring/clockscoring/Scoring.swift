//
//  Scoring.swift
//  clockscoring
//
//  Created by Yi Chong Ong on 9/7/21.
//

import Foundation

class Scoring {
    let OBJ_THRESH = 0.6 // threshold to determine bbox has object
    static let HAND_X_WIDTH = 150 // threshold on how far away mid point of hand must be from contour center before not considered middle
    static let HAND_Y_HEIGHT = 150
    static let LONG_SHORT_THRESH = 5
    
    enum CLASSES : Int, Equatable {
        case ONE = 1
        case TWO = 2
        case THREE = 3
        case FOUR = 4
        case FIVE = 5
        case SIX = 6
        case SEVEN = 7
        case EIGHT = 8
        case NINE = 9
        case TEN = 10
        case ELEVEN = 11
        case TWELEVE = 12
        case CONTOUR = 13
        case HAND = 14
        case TOP_LEFT_ARROW = 15
        case TOP_RIGHT_ARROW = 16
        case RIGHT_ARROW = 17
        case LEFT_ARROW = 18
        case UP_ARROW = 19
        case DOWN_ARROW = 20
        case BOTTOM_RIGHT_ARROW = 21
        case BOTTOM_LEFT_ARROW = 22
    }
    
    // return contour, digit, hand score
    public static func score(predictions : [Prediction], time : Dictionary<String, Int>) -> (Dictionary<String, Int>, Dictionary<String, String>) {
        var scores = [String : Int]()
        var contourScore = 0, digitScore = 0, handScore = 0
        var errorDigitMsg = "", errorHandMsg = ""
        var classDct = convert(predictions: predictions)
        let hasContour = Array(classDct.keys).contains(CLASSES.CONTOUR)
        if hasContour {
            contourScore = 1
        } else {
            classDct = addContour(classDct: classDct)
        }
        
        (digitScore, errorDigitMsg) = scoreDigits(classDct: classDct)
        (handScore, errorHandMsg) = scoreHand(classDct: classDct, time: time)
        scores["contour"] = contourScore; scores["digit"] = digitScore; scores["hand"] = handScore
        
        var errorMsgs = [String : String]()
        if !hasContour {
            errorMsgs["contour"] = "Did not detect clock face"
        }
        if digitScore == 0 {
            errorMsgs["digit"] = errorDigitMsg
        }
        if handScore == 0 {
            errorMsgs["hand"] = errorHandMsg
        }
        return (scores, errorMsgs)
    }
    
    // return score and reason for score if fail
    private static func scoreHand(classDct : Dictionary<CLASSES, BoundingBox>, time : Dictionary<String, Int>) -> (Int, String) {
        var msg = ""
        let hour = time["hour"]!
        var minute = time["minute"]!
        if minute == 0 {
            minute = 12
        } else {
            minute = Int(minute / 5)
        }
        let oneClockHand = (classDct.filter({$0.key == CLASSES.HAND}).count == 1)   // only one clock hand (14) detected
        let correctSubHand = (classDct.keys.filter({$0.rawValue > 14}).count == 2)  // check that there are exactly 2 subhands
        
        if (!oneClockHand || !correctSubHand) {
            msg = "Did not detecct clock hand(14) or two sub hands"
            return (0, msg)
        }
        
        let contourBbox = classDct[CLASSES.CONTOUR]!
        let contourMidX = Int((contourBbox.x1 + contourBbox.x2) / 2), contourMidY = Int((contourBbox.y1 + contourBbox.y2) / 2)
        
        let subHands = Array(classDct.filter({$0.key.rawValue > 14}).keys)
        let subHand1Cls = subHands[0], subHand2Cls = subHands[1]
        let subHand1Bbox = classDct[subHand1Cls]!, subHand2Bbox = classDct[subHand2Cls]!
        let subHand1Dist = hypot(x: Double(subHand1Bbox.x1 - subHand1Bbox.x2), y: Double(subHand1Bbox.y1 - subHand1Bbox.y2)), subHand2Dist = hypot(x: Double(subHand2Bbox.x1 - subHand2Bbox.x2), y: Double(subHand2Bbox.y1 - subHand2Bbox.y2))
        if (fabs(subHand1Dist - subHand2Dist) <= Double(LONG_SHORT_THRESH)) {
            msg = "hands are not distinguishable"
            return (0, msg)
        }
        
        var subHandDct = [String : (CLASSES, BoundingBox)]()
        if subHand1Dist > subHand2Dist {
            subHandDct["long"] = (subHand1Cls, subHand1Bbox)
            subHandDct["short"] = (subHand2Cls, subHand2Bbox)
        } else {
            subHandDct["long"] = (subHand2Cls, subHand2Bbox)
            subHandDct["short"] = (subHand1Cls, subHand1Bbox)
        }
        
        let digitsCenterPos = getDigitCenterPos(classDct: classDct)
        
        let (longHandCls, longHandBbox) = subHandDct["long"]!
        let (shortHandCls, shortHandBbox) = subHandDct["short"]!
        
        // which corner depends on arrow class
        
        let (longHeadX, longHeadY, longTailX, longTailY) = getArrowCoords(arrow_obj: longHandCls, arrowBbox: longHandBbox)
        let (shortHeadX, shortHeadY, shortTailX, shortTailY) = getArrowCoords(arrow_obj: shortHandCls, arrowBbox: shortHandBbox)
        
        // compute distance of arrow head coords to digits
        let tupleArrayLong = digitsCenterPos.map{(key: CLASSES, value: (Int, Int)) in return (key, hypot(x: Double(longHeadX - value.0), y: Double(longHeadY - value.1)))}
        let tupleArrayShort = digitsCenterPos.map{(key: CLASSES, value: (Int, Int)) in return (key, hypot(x: Double(shortHeadX - value.0), y: Double(shortHeadY - value.1)))}
        let longDigitDist = Dictionary(uniqueKeysWithValues: tupleArrayLong)
        let shortDigitDist = Dictionary(uniqueKeysWithValues: tupleArrayShort)
        
        let tailMidX = Int((longTailX + shortTailX) / 2), tailMidY = Int((longTailY + shortTailY) / 2) // for centering
        
        let nearestDigitLongArrow = longDigitDist.min{a, b in a.value < b.value}!.key
        let nearestDigitShortArrow = shortDigitDist.min{a, b in a.value < b.value}!.key
        
        // if mid point of clock hand x and y does not lie within interval
        if (!(contourMidX - (HAND_X_WIDTH / 2) < tailMidX && tailMidX < contourMidX + (HAND_X_WIDTH / 2)) || !(contourMidY - (HAND_Y_HEIGHT / 2) < tailMidY && tailMidY < contourMidY + (HAND_Y_HEIGHT / 2))) {
            msg = "Hand not centered"
            return (0, msg)
        }
        
        if nearestDigitLongArrow.rawValue == minute && nearestDigitShortArrow.rawValue == hour {
            msg = "passed"
            return (1, msg)
        } else {
            msg = "long or short hand not pointing to correct digit"
            return (0, msg)
        }
    }
    
    // return score and reason for score if fail
    private static func scoreDigits(classDct : Dictionary<CLASSES, BoundingBox>) -> (Int, String) {
        var msg = ""
        var digits = classDct.keys.filter{$0.rawValue < 13}
        print(digits)
        if digits.count != 12 {
            msg = "did not detect all 12 digits"
            return (0, msg)
        }
        let digitsCenterPos = getDigitCenterPos(classDct: classDct)
        let highestDigitY = digitsCenterPos.min(by: {a, b in a.value.1 < b.value.1})!.key
        if highestDigitY != CLASSES.TWELEVE {
            msg = "Digit 12 not in correct order"
            return (0, msg)
        }
        let contourBbox = classDct[CLASSES.CONTOUR]!
        let contourMidX = Int((contourBbox.x1 + contourBbox.x2) / 2), contourMidY = Int((contourBbox.y1 + contourBbox.y2) / 2)
        digits.sort(by: {(a: CLASSES, b: CLASSES) in a.rawValue < b.rawValue})
        let tweleveRef = digits[digits.count - 1]
        let tweleveRefCoord = digitsCenterPos[tweleveRef]!
        var angles = [Double]()
        var startVec = [tweleveRefCoord.0 - contourMidX, tweleveRefCoord.1 - contourMidY]
        var cumSum = 0.0
        for digit in digits {
            let digitCoord = digitsCenterPos[digit]!
            let digitVec = [digitCoord.0 - contourMidX, tweleveRefCoord.1 - contourMidY]
            let angle = computeAngle(vec1: startVec, vec2: digitVec)
            startVec = digitVec
            cumSum += angle
            angles.append(cumSum)
        }
        if (angles.filter({$0 > 361}).isEmpty) {
            msg = "passed check"
            return (1, msg)
        }
        return (0, "digits not in correct order")
    }
    
    // convert a list of predictions to dictionary of classes : boundingBox
    private static func convert(predictions: [Prediction]) -> Dictionary<CLASSES, BoundingBox> {
        var classDict = [CLASSES: BoundingBox]()
        for prediction in predictions {
            let label : Int = prediction.classIndex
            let bbox : BoundingBox = prediction.bbox
            let classIdx : CLASSES = CLASSES(rawValue: label)!
            classDict[classIdx] = bbox
        }
        return classDict
    }
    
    // return head and tail coordinates respectively
    private static func getArrowCoords(arrow_obj : CLASSES, arrowBbox : BoundingBox) -> (Int, Int, Int, Int){
        switch arrow_obj {
        case .TOP_LEFT_ARROW:
            let topLeftX = arrowBbox.x1
            let topLeftY = arrowBbox.y1
            let tailX = arrowBbox.x2
            let tailY = arrowBbox.y2
            return (topLeftX, topLeftY, tailX, tailY)
        case .TOP_RIGHT_ARROW:
            let topRightX = arrowBbox.x2
            let topRightY = arrowBbox.y1
            let tailX = arrowBbox.x1
            let tailY = arrowBbox.y2
            return (topRightX, topRightY, tailX, tailY)
        case .RIGHT_ARROW:
            let topRightX = arrowBbox.x2
            let topRightY = arrowBbox.y1
            // let botRightX = arrowBbox.x2
            let botRightY = arrowBbox.y2
            let tailX = arrowBbox.x1
            return (topRightX, Int((topRightY + botRightY) / 2 ), tailX, Int((topRightY + botRightY) / 2))
        case .LEFT_ARROW:
            let topLeftX = arrowBbox.x1
            let topLeftY = arrowBbox.y1
            // let botLeftX = arrowBbox.x1
            let botLeftY = arrowBbox.y2
            let tailX = arrowBbox.x2
            return (topLeftX, Int((topLeftY + botLeftY) / 2), tailX, Int((topLeftY + botLeftY) / 2))
        case .UP_ARROW:
            let topLeftX = arrowBbox.x1
            let topLeftY = arrowBbox.y1
            let topRightX = arrowBbox.x2
            // let topRightY = arrowBbox.y1
            let tailY = arrowBbox.y2
            return (Int((topLeftX + topRightX) / 2), topLeftY, Int((topLeftX + topRightX) / 2), tailY)
        case .DOWN_ARROW:
            let botLeftX = arrowBbox.x1
            let botLeftY = arrowBbox.y2
            let botRightX = arrowBbox.x2
            // let botRightY = arrowBbox.y2
            let tailY = arrowBbox.y1
            return (Int((botLeftX + botRightX) / 2), botLeftY, Int((botLeftX + botRightX) / 2), tailY)
        case .BOTTOM_RIGHT_ARROW:
            let botRightX = arrowBbox.x2
            let botRightY = arrowBbox.y2
            let tailX = arrowBbox.x1
            let tailY = arrowBbox.y1
            return (botRightX, botRightY, tailX, tailY)
        case .BOTTOM_LEFT_ARROW:
            let botLeftX = arrowBbox.x1
            let botLeftY = arrowBbox.y2
            let tailX = arrowBbox.x2
            let tailY = arrowBbox.y1
        return (botLeftX, botLeftY, tailX, tailY)
        default:
            return (0, 0, 0, 0)
        }
    }
    
    // both vec1 and vec2 same length
    private static func hypot(x : Double, y : Double) -> Double {
        return (x * x + y * y).squareRoot()
    }
    
    // get center positions of digits
    private static func getDigitCenterPos(classDct : Dictionary<CLASSES, BoundingBox>) -> Dictionary<CLASSES, (Int, Int)> {
        let digits = 1..<13
        var digitsCenterPos = [CLASSES : (Int, Int)]()
        for (key, value) in classDct {
            if digits.contains(key.rawValue) {
                let digitBbox = value
                let digitX1 = digitBbox.x1, digitY1 = digitBbox.y1, digitX2 = digitBbox.x2, digitY2 = digitBbox.y2
                let digitCenterX = Int((digitX1 + digitX2) / 2), digitCenterY = Int((digitY1 + digitY2) / 2)
                digitsCenterPos[key] = (digitCenterX, digitCenterY)
            }
        }
        return digitsCenterPos
    }
    
    // 
    private static func computeAngle(vec1 : [Int], vec2: [Int]) -> Double {
        let dotProd = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        let vec1Mag = hypot(x: Double(vec1[0]), y: Double(vec1[1]))
        let vec2Mag = hypot(x: Double(vec2[0]), y: Double(vec2[1]))
        let cosAngle = Double(dotProd) / (vec1Mag * vec2Mag)
        let angle = acos(cosAngle)
        return (angle * (180 / Double.pi))
    }
    
    private static func addContour(classDct : Dictionary<CLASSES, BoundingBox>) -> Dictionary<CLASSES, BoundingBox> {
        // get all digit Bboxes
        var newDict = classDct
        let digitBbox = classDct.filter {$0.key.rawValue < 13}
        let filteredBbox = digitBbox.values
        
        func mapX1(box : BoundingBox) -> Int {
            return box.x1
        }
        
        func mapX2(box : BoundingBox) -> Int {
            return box.x2
        }
        
        func mapY1(box : BoundingBox) -> Int {
            return box.y1
        }
        
        func mapY2(box : BoundingBox) -> Int {
            return box.y2
        }
        
        let xmin = filteredBbox.map(mapX1).min()!
        let xmax = filteredBbox.map(mapX2).max()!
        let ymin = filteredBbox.map(mapY1).min()!
        let ymax = filteredBbox.map(mapY2).max()!
        
        newDict[CLASSES.CONTOUR] = BoundingBox(x1: xmin, y1: ymin, x2: xmax, y2: ymax)
        
        return newDict
        
    }
    
}
