#include "ObjectDetector.h"

namespace TATS {
	ObjectDetector::ObjectDetector()
	{
	}

	// 0 is color; 1 is gray. Grey by default.
	void ObjectDetector::setInputColor(int code)
	{
		if (code <= 1) {
			input_color = code;
		}
	}
};


