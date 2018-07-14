def get_landmarks(im):
    # Returns facial landmarks as (x,y) coordinates
    rects = detector(im, 1)
    
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
