Screenshooting
==============

The main purpose of these scripts was to automatically get working hours out of them because I am too lazy to fill them by hand. I can do this quite easily as for one job, I mostly must use Eclipse. And I don't use Eclipse for anything else. And I don't have Eclipse in my Dock if it is not running. So I 'just' have to check a screenshot if it has the Eclipse icon in the Dock and I know that I was working for that job.

Another purpose of these scripts was to get a bit used to OpenCV.

a bunch of scripts
------------------

* `screener.sh`: creates screenshot every 10 sec
* `find-dock.sh`: finds the mac dock and its icons in existing screenshots
* `find-eclipse.*`: finds Eclipse icon in dock in existing screenshots
* ...

Development history
-------------------

The first thing I tried were to adopt some of the OpenCV demos and technics to my use case -- without having any background knowledge about the technics and neither understand what they are and how they work.

One of these demos were face recognition done with Haar classification. I tried to train the classifier to detect the Eclipse icon but failed. I am still not exactly sure if I had chosen bad parameters at the training or did something else wrongly.

Then I tried some of the other classifiers but I failed to make them find the icon in the screenshot.

Then I decided to dive a bit into the theory. I bought the book about OpenCV and started reading. After reading most of it roughly, I know have a much better understanding about what all the technics are and how they work.

I tried again with doing another Haar training with better parameters. Though, it seemed that now the training itself never really finished (or my machine is just too weak). So I also gave up on that again.

Then I decided to try a more manual and explicit approach. With some clever  color histogram checks, I tried to determine where the Dock is exactly. After some hacking, this worked more or less. Then I tried to detect a list of icons in the Dock. I used some similar tricky histogram checks here. This was a bit harder though and my solution was even less accurate. But it mostly worked. Then I generalized all that code so that it could find the Dock no matter if it was in the bottom, left or right.

Based on this, I wrote another script which now iterates through the list of all icons (i.e. rectangles on the screenshot) and checked each one if it matches the Eclipse icon. This was also much harder than I thought initially. Just comparing the color histogram was not good enough. Just using the Euclidean distance also didn't really worked that well. I again came up with an own solution which basically moved or scaled the Eclipse icon in about 100 different ways, compared the Euclidean distance each time and returned the minimum. Also it ignored all white pixels in the Eclipse icon because that was the background color (initially, I wanted to use the alpha channel but OpenCV, while loading it and even having a fourth channel on all loaded images, it resets the alpha channel for some reason). This solution mostly worked.

All put together worked more or less. It was terribly slow because in each step, I iterated through a really huge set of possibilities. Also, I had a quite high rate of both false positives and false negatives. But it was anyway so slow (about 30-60 seconds for one screenshot) that it wasn't really useable for me.

Then I decided to again start with another idea from scratch. I took 16 representative colors from the Eclipse icon and searched for areas with similar colors in the screenshot. I could use cvMatchTemplate for that. My first code which did this was ready in just within 10 minutes or so and I was shocked at how good that was. In about 0.5 seconds, I got about 5 spots or so on a screenshot and the Eclipse icon was always one of it. Then, I extended this by some more handling which tried to make good rectangles over the found spots and filtering those out which are not possibly Dock icons. That was a bit harder and more tricky now but I got it working really well. Then I had to do a final check if the found rectangles matched the Eclipse icon. This was the old problem I already had earlier. I just reused the best solution I had come up with earlier for this. And put all together, I now have a working solution (which seems to be about 99% or even more accurate) which can handle one screenshot in about 2-5 seconds.

So there I am. This is my final solution for this problem now.

My experiences with OpenCV are very mixed. It is very powerful without doubt. But it seems very inconsistent at many places and also has some serious bugs. E.g. there is a serious memo leak when you create a new histogram; for that reason, I had to hack a histogram pool where it reused existing histograms. Also many functions are somewhat unnatural to use: If you want to get a pixel from an image, it is cv.Get2D(image, y, x). If you create a histogram, it is uninitialized. And so on. That introduced bugs which I haven't really thought about and cost some nerves finding them. Also, the OpenCV IRC channel is quite dead and the only other way to get in touch with other users is through a Yahoo group. They also don't really respond very fast to bug reports (well, they haven't at all yet to any of my reports).

I will watch OpenCV in the future a bit more closely and see how it develops.


-- Albert Zeyer, [www.az2000.de](http://www.az2000.de)
