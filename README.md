# Hough-Transform-on-Ellipses
Implementing a research paper algorithm (by H.K. Yuen et all,. 1988) for ellipse detector via HT + Preprocessing pipeline

Research Paper: ELLIPSE DETECTION USING THE HOUGH TRANSFORM (H.K. Yuen, J. Illingworth and J. Kittler., 1988)
Research Paper link: http://www.bmva.org/bmvc/1988/avc-88-041.pdf

Finding an ellipse, despite being a well configured task, consists in a few complicated and intricate smaller labors. Part of are curvature, filtering, finding edges as well as masks, different algorithms and using some mathematical shape-oriented operations.
The implemented algorithm offers a pre-processing pipeline for calculating the image edge map, by emphasizing the ellipse curvature and edge points. As in addition calculates the suspected ellipses point's tangent and its slope, calculates the TM lines in a smart manner and calculates the ellipse hypothesis center by encountering some of the points, and pairing them, in a controllable manner.

Results of applying the above on different variety of images are attached in this folder as well as a final report.
