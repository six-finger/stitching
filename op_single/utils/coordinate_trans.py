import math
import numpy as np

pi = math.pi 
d2r = pi/180
r2d = 180/pi

a = 6378137
f = 1/298.257223563
b = a*(1 - f)
ee = f*(2-f)    # e = sqrt(2*f-f**2); ee=1-b**2/a**2
epsilon = 0.000000000000001
BLH0 = np.array([38,113,0])

def deg2rad(a):
    return a*math.pi/180
def cos(a):
    return math.cos(a)
def sin(a):
    return math.sin(a)
def tan(a):
    return math.tan(a)
def atan2(a,b):
    return math.atan2(a,b)
def sqrt(a):
    return math.sqrt(a)

def BLH2XYZ(BLH):
    B,L,H = BLH
    B,L,H = B*d2r, L*d2r, H*d2r
    N = a / sqrt(1 - ee * sin(B)**2)
    
    X = (N+H)*cos(B)*cos(L)
    Y = (N+H)*cos(B)*sin(L)
    Z = (N*(1-ee)+H)*sin(B)
    return np.array([X,Y,Z])

def XYZ2BLH(XYZ):
    X,Y,Z = XYZ
    curB = 0
    calB = atan2(Z, sqrt(X**2 + Y**2))
    cnt = 0
    n = 0
    while abs(curB - calB)*r2d > epsilon and cnt < 25:
        curB = calB
        n = a / sqrt(1 - ee * sin(curB)**2)
        calB = atan2(Z + n*ee*sin(curB), sqrt(X**2 + Y**2))
        cnt += 1
    B = curB * r2d
    L = atan2(Y, X) * r2d
    H = (Z / sin(curB) - n*(1 - ee)) * r2d
    return np.array([B,L,H])
    
def XYZ2ENU(XYZ, B0L0H0=BLH0):
    X,Y,Z = XYZ
    B0,L0,H0 = B0L0H0
    B0,L0,H0 = B0*d2r, L0*d2r, H0*d2r
    N = a / sqrt(1 - ee * sin(B0)**2)

    X0 = (N+H0)*cos(B0)*cos(L0)
    Y0 = (N+H0)*cos(B0)*sin(L0)
    Z0 = (N*(1-ee)+H0)*sin(B0)
    Xd = X - X0
    Yd = Y - Y0
    Zd = Z - Z0
    
    east  = -        sin(L0)*Xd +         cos(L0)*Yd
    north = -sin(B0)*cos(L0)*Xd - sin(B0)*sin(L0)*Yd + cos(B0)*Zd
    up    =  cos(B0)*cos(L0)*Xd + cos(B0)*sin(L0)*Yd + sin(B0)*Zd
    return np.array([east, north, up])
    
def ENU2XYZ(ENU, B0L0H0=BLH0):
    east, north, up = ENU
    B0,L0,H0 = B0L0H0
    B0,L0,H0 = B0*d2r, L0*d2r, H0*d2r
    N = a / sqrt(1 - ee * sin(B0)**2)
    
    X0 = (N+H0)*cos(B0)*cos(L0)
    Y0 = (N+H0)*cos(B0)*sin(L0)
    Z0 = (N*(1-ee)+H0)*sin(B0)
    
    Xd = -sin(L0)*east - sin(B0)*cos(L0)*north + cos(B0)*cos(L0)*up
    Yd =  cos(L0)*east - sin(B0)*sin(L0)*north + cos(B0)*sin(L0)*up
    Zd =        0*east +         cos(B0)*north +         sin(B0)*up

    X = Xd + X0
    Y = Yd + Y0
    Z = Zd + Z0
    return np.array([X,Y,Z])

def BLH2ENU(BLH, B0L0H0=BLH0):    
    XYZ = BLH2XYZ(BLH)
    ENU = XYZ2ENU(XYZ, B0L0H0)
    return ENU
    
def ENU2BLH(ENU, B0L0H0=BLH0):
    XYZ = ENU2XYZ(ENU, B0L0H0)
    BLH = XYZ2BLH(XYZ)
    return BLH

def dist(XYZ1, XYZ2):
    arr = XYZ1 - XYZ2
    d = sqrt(arr[0]**2 + arr[1]**2 + arr[2]**2)
    return d


if __name__ == '__main__':
    # ----------------------
    # ZISUO
    # measurement
    BLH5_m = np.array([39.9779251138889, 116.326277819444, 51.0687])
    BLH6_m = np.array([39.9780045638889, 116.326257680556, 51.0015])
    BLH7_m = np.array([39.9780673916667, 116.326492047222, 50.9103])
    BLH8_m = np.array([39.9788387194444, 116.326371763889, 50.9058])
    BLH9_m = np.array([39.9780114861111, 116.325825836111, 50.7557])
    BLH10_m = np.array([39.97779285, 116.3258354, 50.7969])
    # google
    BLH5_g = np.array([39.9779231, 116.3262764, 51.0687])
    BLH6_g = np.array([39.9780038, 116.326257, 51.0015])
    BLH7_g = np.array([39.9780701, 116.3264897, 50.9103])
    BLH8_g = np.array([39.9788378, 116.3263731, 50.9058])
    BLH9_g = np.array([39.9780121, 116.3258286, 50.7557])
    BLH10_g = np.array([39.9777932, 116.325836, 50.7969])

    # print(dist(BLH2XYZ(BLH5_m), BLH2XYZ(BLH5_g)))
    # print(dist(BLH2XYZ(BLH6_m), BLH2XYZ(BLH6_g)))
    # print(dist(BLH2XYZ(BLH7_m), BLH2XYZ(BLH7_g)))
    # print(dist(BLH2XYZ(BLH8_m), BLH2XYZ(BLH8_g)))
    # print(dist(BLH2XYZ(BLH9_m), BLH2XYZ(BLH9_g)))
    # print(dist(BLH2XYZ(BLH10_m), BLH2XYZ(BLH10_g)))

    # -------------------------------------------------
    # WEIHAI
    # GPS in google
    BLH_111_G = np.array([37.5475171, 122.0886264, 50])
    BLH_112_G = np.array([37.5478381, 122.0882335, 50])
    BLH_114_G = np.array([37.5484229, 122.0876473, 50])
    BLH_115_G = np.array([37.5486423, 122.0873458, 50])
    BLH_116_G = np.array([37.5487889, 122.0871066, 50])
    BLH_121_G = np.array([37.5473578, 122.0850364, 50])
    BLH_135_G = np.array([37.5434525, 122.0895372, 50])
    BLH_136_G = np.array([37.5434443, 122.0896044, 50])
    BLH_157_G = np.array([37.5478466, 122.0865492, 50])
    BLH_161_G = np.array([37.5466652, 122.0876904, 50])
    
    # GPS of measurement resultsd
    BLH_111_M = np.array([37.5475099,122.0888489, 50])
    BLH_112_M = np.array([37.5478343,122.0885801, 50])
    BLH_114_M = np.array([37.5483506,122.0878983, 50])
    BLH_115_M = np.array([37.5486048, 122.0876193, 50])
    BLH_116_M = np.array([37.5488921,122.0873503, 50])
    BLH_121_M = np.array([37.5474449,122.0852128, 50])
    BLH_135_M = np.array([37.5435497, 122.0896367, 50])
    BLH_136_M = np.array([37.5434807, 122.0898171, 50])
    BLH_157_M = np.array([37.5479258, 122.0867076, 50])
    BLH_161_M = np.array([37.5467728, 122.0878752, 50])
    
    # GPS of UAV
    BLH_115_UAV = np.array([37.5485124,122.0875981,50])
    BLH_121_UAV = np.array([37.54754974,122.0850909,50])
    BLH_135_UAV = np.array([37.54380989,122.0894760,50])
    BLH_136_UAV = np.array([37.54354091,122.0897903,50])
    BLH_157_UAV = np.array([37.54797238,122.0864643,50])
    BLH_161_UAV = np.array([37.54691912,122.0877186,50])

    
    # print(dist(BLH2XYZ(BLH_115_G), BLH2XYZ(BLH_115_M)))
    # print(dist(BLH2XYZ(BLH_121_G), BLH2XYZ(BLH_121_M)))
    # print(dist(BLH2XYZ(BLH_135_G), BLH2XYZ(BLH_135_M)))
    # print(dist(BLH2XYZ(BLH_136_G), BLH2XYZ(BLH_136_M)))
    # print(dist(BLH2XYZ(BLH_157_G), BLH2XYZ(BLH_157_M)))
    # print(dist(BLH2XYZ(BLH_161_G), BLH2XYZ(BLH_161_M)))

    # print(BLH_115_G - BLH_115_UAV, BLH_115_G - BLH_115_M)
    # print(BLH_121_G - BLH_121_UAV, BLH_121_G - BLH_121_M)
    # print(BLH_135_G - BLH_135_UAV, BLH_135_G - BLH_135_M)
    # print(BLH_136_G - BLH_136_UAV, BLH_136_G - BLH_136_M)
    # print(BLH_157_G - BLH_157_UAV, BLH_157_G - BLH_157_M)
    # print(BLH_161_G - BLH_161_UAV, BLH_161_G - BLH_161_M)
    
    offset = []
    offset.append(BLH2ENU(BLH_111_G) - BLH2ENU(BLH_111_M))
    offset.append(BLH2ENU(BLH_112_G) - BLH2ENU(BLH_112_M))
    # offset.append(BLH2ENU(BLH_114_G) - BLH2ENU(BLH_114_M))
    # offset.append(BLH2ENU(BLH_115_G) - BLH2ENU(BLH_115_M))
    # offset.append(BLH2ENU(BLH_116_G) - BLH2ENU(BLH_116_M))
    # offset.append(BLH2ENU(BLH_121_G) - BLH2ENU(BLH_121_M))
    # offset.append(BLH2ENU(BLH_135_G) - BLH2ENU(BLH_135_M))
    # offset.append(BLH2ENU(BLH_136_G) - BLH2ENU(BLH_136_M))
    offset.append(BLH2ENU(BLH_157_G) - BLH2ENU(BLH_157_M))
    offset.append(BLH2ENU(BLH_161_G) - BLH2ENU(BLH_161_M))
    for i in offset:
        print(i)
    offset = np.array(offset)
    offset = np.array([np.median(offset[:, 0]), np.median(offset[:, 1]), np.median(offset[:, 2])])
    # print("offset: ", offset)
        
    # print(BLH_115_G - XYZ2BLH(BLH2ENU(BLH_115_M)-offset))
    # print(BLH_121_G - XYZ2BLH(BLH2ENU(BLH_121_M)-offset))
    # print(BLH_135_G - XYZ2BLH(BLH2ENU(BLH_135_M)-offset))
    # print(BLH_136_G - XYZ2BLH(BLH2ENU(BLH_136_M)-offset))
    # print(BLH_157_G - XYZ2BLH(BLH2ENU(BLH_157_M)-offset))
    # print(BLH_161_G - XYZ2BLH(BLH2ENU(BLH_161_M)-offset))
    
    print(dist(BLH2ENU(BLH_111_G), BLH2ENU(BLH_111_M)+offset))
    print(dist(BLH2ENU(BLH_112_G), BLH2ENU(BLH_112_M)+offset))
    print(dist(BLH2ENU(BLH_114_G), BLH2ENU(BLH_114_M)+offset))
    print(dist(BLH2ENU(BLH_115_G), BLH2ENU(BLH_115_M)+offset))
    print(dist(BLH2ENU(BLH_116_G), BLH2ENU(BLH_116_M)+offset))
    
    print(dist(BLH2ENU(BLH_121_G), BLH2ENU(BLH_121_M)+offset))
    print(dist(BLH2ENU(BLH_135_G), BLH2ENU(BLH_135_M)+offset))
    print(dist(BLH2ENU(BLH_136_G), BLH2ENU(BLH_136_M)+offset))
    print(dist(BLH2ENU(BLH_157_G), BLH2ENU(BLH_157_M)+offset))
    print(dist(BLH2ENU(BLH_161_G), BLH2ENU(BLH_161_M)+offset))
