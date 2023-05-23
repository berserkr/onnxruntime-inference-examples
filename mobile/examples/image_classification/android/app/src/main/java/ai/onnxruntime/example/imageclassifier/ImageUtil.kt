/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.onnxruntime.example.imageclassifier

import android.graphics.*
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer

const val DIM_BATCH_SIZE = 1;
const val DIM_PIXEL_SIZE = 3;
//const val IMAGE_SIZE_X = 320; // default is 224
//const val IMAGE_SIZE_Y = IMAGE_SIZE_X;
//val IMAGENET_PIXEL_MEAN = {123.675, 116.280, 103.530};
//val IMAGENET_PIXEL_STD = {58.395, 57.12, 57.375};

fun preProcess(bitmap: Bitmap, resolution: Int): FloatBuffer {
    val imgData = FloatBuffer.allocate(
            DIM_BATCH_SIZE
                    * DIM_PIXEL_SIZE
                    * resolution
                    * resolution
    )
    imgData.rewind()
    val stride = resolution * resolution
    val bmpData = IntArray(stride)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    for (i in 0..resolution - 1) {
        for (j in 0..resolution - 1) {
            val idx = resolution * i + j
            val pixelValue = bmpData[idx]

            val red = Color.red(pixelValue)
            val green = Color.green(pixelValue)
            val blue = Color.blue(pixelValue)

//            Log.i(MainActivity.TAG, "Red: " + red + " => (" + idx + ", " + (red  / (red - 123.675f) / 58.395f) + ")")
//            Log.i(MainActivity.TAG, "Green: " + green + " => (" + idx + ", " + (green  / (green - 116.280f) / 57.12f) + ")")
//            Log.i(MainActivity.TAG, "Blue " + blue + " => (" + idx + ", " + (blue  / (blue - 103.530f) / 57.375f) + ")")

//            imgData.put(idx * 3, (((red) / 255f - 0.485f) / 0.229f))
//            imgData.put(idx * 3 + 1, (((green) / 255f - 0.456f) / 0.224f))
//            imgData.put(idx * 3 + 2, (((blue) / 255f - 0.406f) / 0.225f))

//            imgData.put(idx, (((red) / 255f - 0.485f) / 0.229f))
//            imgData.put(idx + stride, (((green) / 255f - 0.456f) / 0.224f))
//            imgData.put(idx + stride * 2, (((blue) / 255f - 0.406f) / 0.225f))

            imgData.put(idx, (red - 123.675f) / 58.395f)
            imgData.put(idx + stride, ( green - 116.280f) / 57.12f)
            imgData.put(idx + stride * 2, (blue - 103.530f) / 57.375f)
        }
    }

    imgData.rewind()
    return imgData
}

fun ImageProxy.toBitmap(): Bitmap? {
    val nv21 = yuv420888ToNv21(this)
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    return yuvImage.toBitmap()
}

private fun YuvImage.toBitmap(): Bitmap? {
    val out = ByteArrayOutputStream()
    if (!compressToJpeg(Rect(0, 0, width, height), 100, out))
        return null
    val imageBytes: ByteArray = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
    val pixelCount = image.cropRect.width() * image.cropRect.height()
    val pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888)
    val outputBuffer = ByteArray(pixelCount * pixelSizeBits / 8)
    imageToByteBuffer(image, outputBuffer, pixelCount)
    return outputBuffer
}

private fun imageToByteBuffer(image: ImageProxy, outputBuffer: ByteArray, pixelCount: Int) {
    assert(image.format == ImageFormat.YUV_420_888)

    val imageCrop = image.cropRect
    val imagePlanes = image.planes

    imagePlanes.forEachIndexed { planeIndex, plane ->
        // How many values are read in input for each output value written
        // Only the Y plane has a value for every pixel, U and V have half the resolution i.e.
        //
        // Y Plane            U Plane    V Plane
        // ===============    =======    =======
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        val outputStride: Int

        // The index in the output buffer the next value will be written at
        // For Y it's zero, for U and V we start at the end of Y and interleave them i.e.
        //
        // First chunk        Second chunk
        // ===============    ===============
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        var outputOffset: Int

        when (planeIndex) {
            0 -> {
                outputStride = 1
                outputOffset = 0
            }
            1 -> {
                outputStride = 2
                // For NV21 format, U is in odd-numbered indices
                outputOffset = pixelCount + 1
            }
            2 -> {
                outputStride = 2
                // For NV21 format, V is in even-numbered indices
                outputOffset = pixelCount
            }
            else -> {
                // Image contains more than 3 planes, something strange is going on
                return@forEachIndexed
            }
        }

        val planeBuffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        // We have to divide the width and height by two if it's not the Y plane
        val planeCrop = if (planeIndex == 0) {
            imageCrop
        } else {
            Rect(
                    imageCrop.left / 2,
                    imageCrop.top / 2,
                    imageCrop.right / 2,
                    imageCrop.bottom / 2
            )
        }

        val planeWidth = planeCrop.width()
        val planeHeight = planeCrop.height()

        // Intermediate buffer used to store the bytes of each row
        val rowBuffer = ByteArray(plane.rowStride)

        // Size of each row in bytes
        val rowLength = if (pixelStride == 1 && outputStride == 1) {
            planeWidth
        } else {
            // Take into account that the stride may include data from pixels other than this
            // particular plane and row, and that could be between pixels and not after every
            // pixel:
            //
            // |---- Pixel stride ----|                    Row ends here --> |
            // | Pixel 1 | Other Data | Pixel 2 | Other Data | ... | Pixel N |
            //
            // We need to get (N-1) * (pixel stride bytes) per row + 1 byte for the last pixel
            (planeWidth - 1) * pixelStride + 1
        }

        for (row in 0 until planeHeight) {
            // Move buffer position to the beginning of this row
            planeBuffer.position(
                    (row + planeCrop.top) * rowStride + planeCrop.left * pixelStride
            )

            if (pixelStride == 1 && outputStride == 1) {
                // When there is a single stride value for pixel and output, we can just copy
                // the entire row in a single step
                planeBuffer.get(outputBuffer, outputOffset, rowLength)
                outputOffset += rowLength
            } else {
                // When either pixel or output have a stride > 1 we must copy pixel by pixel
                planeBuffer.get(rowBuffer, 0, rowLength)
                for (col in 0 until planeWidth) {
                    outputBuffer[outputOffset] = rowBuffer[col * pixelStride]
                    outputOffset += outputStride
                }
            }
        }
    }
}