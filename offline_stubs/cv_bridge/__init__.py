class CvBridge:
    def imgmsg_to_cv2(self, *a, **k):
        raise NotImplementedError('cv_bridge stub: raw Image path unavailable offline')
    def cv2_to_imgmsg(self, *a, **k):
        raise NotImplementedError('cv_bridge stub')
    def cv2_to_compressed_imgmsg(self, *a, **k):
        raise NotImplementedError('cv_bridge stub')
