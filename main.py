import imagemanip

CONST_FILE = 'mandrill.jpg'

CONST_SIGMA = 5.0
CONST_KWINDOW = 15
CONST_THI = .3
CONST_TLO = .15
CONST_CWINDOW = 23

imagemanip.procAndWrite(CONST_FILE, CONST_SIGMA, CONST_KWINDOW,
                        CONST_THI, CONST_TLO, CONST_CWINDOW)

