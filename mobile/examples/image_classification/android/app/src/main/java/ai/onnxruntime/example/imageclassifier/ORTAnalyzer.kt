// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.*
import kotlin.collections.HashMap
import kotlin.math.exp


internal data class Result(
        var detectedIndices: List<Int> = emptyList(),
        var detectedScore: MutableList<Float> = mutableListOf<Float>(),
        var processTimeMs: Long = 0,
        var peakMemory: Long = 0
) {}

internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val callBack: (Result) -> Unit,
        private val enableQuantizedModel: Boolean
) : ImageAnalysis.Analyzer {

    // Get index of top 3 values
    // This is for demo purpose only, there are more efficient algorithms for topK problems
    private fun getTop3(labelVals: FloatArray): List<Int> {
        var indices = mutableListOf<Int>()
        for (k in 0..2) {
            var max: Float = 0.0f
            var idx: Int = 0
            for (i in 0..labelVals.size - 1) {
                val label_val = labelVals[i]
                if (label_val > max && !indices.contains(i)) {
                    max = label_val
                    idx = i
                }
            }

            indices.add(idx)
        }

        return indices.toList()
    }

    // Calculate the SoftMax for the input array
    private fun softMax(modelResult: FloatArray): FloatArray {
        val labelVals = modelResult.copyOf()
        val max = labelVals.max()
        var sum = 0.0f

        // Get the reduced sum
        for (i in labelVals.indices) {
            labelVals[i] = exp(labelVals[i] - max!!)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }

    // Rotate the image of the input bitmap
    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    override fun analyze(image: ImageProxy) {
        // Convert the input image to bitmap and resize to 224x224 for model input

        val resolution = if (enableQuantizedModel) 192 else 288
        //val resolution = 288
        val imgBitmap = image.toBitmap()
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, resolution, resolution, false) }
        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())

        if (bitmap != null) {
            var result = Result()

//            val it = ortSession?.inputNames?.iterator()
//            while(it?.hasNext()==true) {
//                Log.i(MainActivity.TAG, "Inputs: " + it.next())
//            }

            val imgData = preProcess(bitmap, resolution)
            val it = ortSession?.inputNames?.iterator()
            val inputName = it?.next()

            // needed for yolov3
            //val inputName2 = it?.next()

            val shape = longArrayOf(1, 3, resolution.toLong(), resolution.toLong())
            val env = OrtEnvironment.getEnvironment()
            env.use {
                val tensor = OnnxTensor.createTensor(env, imgData, shape)
                    //Log.i(MainActivity.TAG, "Tensor values: " + tensor.getValue())

                val startTime = SystemClock.uptimeMillis()
                tensor.use {
                    // default:
                    // val output = ortSession?.run(Collections.singletonMap(inputName, tensor))
                    var hashMap : HashMap<String, OnnxTensor>
                            = HashMap<String, OnnxTensor> ()
                    hashMap.put(inputName as String, tensor)

                    //TODO: needed for yolov3
                    /*
                    val tensorAsBuffer : FloatBuffer = FloatBuffer.allocate(2)
                    //tensorAsBuffer.put(0, 1f)
                    //tensorAsBuffer.put(1, 3f)
                    tensorAsBuffer.put(0, 320f)
                    tensorAsBuffer.put(1, 320f)

                    val shapeTensor = OnnxTensor.createTensor(env, tensorAsBuffer, longArrayOf(1, 2))
                    hashMap.put(inputName2 as String, shapeTensor)
                    */


                    val output = ortSession?.run(hashMap)
                    output.use {
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime
                        @Suppress("UNCHECKED_CAST")

                        //Log.i(MainActivity.TAG, "XXX: " + output?.size())

                        // for tofa classification, this works:
                        val rawOutput = ((output?.get(0)?.value) as FloatArray) //
                        //Log.i(MainActivity.TAG, "XXX: " + rawOutput.size)

                        //for (value in rawOutput) {
                        //    Log.i(MainActivity.TAG, "YYY: " + value)
                        //}

                        // for tofa object detection and yolov3
                        //val rawOutput = ((output?.get(0)?.value) as Array<Array<FloatArray>>)[0][0]

                        // default
                        //val rawOutput = ((output?.get(0)?.value) as Array<FloatArray>)[0]

                        val probabilities = softMax(rawOutput)
                        //for(prob in probabilities) {
                        //    Log.i(MainActivity.TAG, "ZZZ: " + prob)
                        //}

                        result.detectedIndices = getTop3(probabilities)
                        for (idx in result.detectedIndices) {
                            result.detectedScore.add(probabilities[idx])
                        }
                    }
                }
            }

            val nativeHeapSize = Debug.getNativeHeapSize()
            val nativeHeapFreeSize = Debug.getNativeHeapFreeSize()
            result.peakMemory = (nativeHeapSize - nativeHeapFreeSize) / (1024*1024) // in MB

            callBack(result)
        }

        image.close()
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}