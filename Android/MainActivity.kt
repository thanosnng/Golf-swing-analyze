package com.programminghut.pose_detection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import com.programminghut.pose_detection.ml.LiteModelMovenetSingleposeLightningTfliteFloat164
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

// MainActivity 클래스는 AppCompatActivity를 상속받아 안드로이드 앱의 메인 활동을 정의합니다.
class MainActivity : AppCompatActivity() {
    // 그래픽스 관련 객체들을 초기화합니다.
    private val paint = Paint()  // 그리기 도구로서 원을 그릴 때 사용됩니다.
    private lateinit var imageProcessor: ImageProcessor  // 이미지 처리를 위한 TensorFlow Lite의 ImageProcessor
    private lateinit var model: LiteModelMovenetSingleposeLightningTfliteFloat164  // TensorFlow Lite 모델
    private lateinit var bitmap: Bitmap  // 카메라로부터 가져온 이미지를 저장할 비트맵
    private lateinit var imageView: ImageView  // UI에서 이미지를 보여줄 ImageView
    private lateinit var handler: Handler  // 백그라운드 스레드에서 작업을 스케줄링하기 위한 핸들러
    private lateinit var handlerThread: HandlerThread  // 백그라운드에서 실행될 스레드
    private lateinit var textureView: TextureView  // 카메라 프리뷰를 보여줄 TextureView
    private lateinit var cameraManager: CameraManager  // 카메라 관리를 위한 시스템 서비스

    // 앱이 시작될 때 호출되는 메서드입니다.
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)  // 레이아웃을 설정합니다.
        get_permissions()  // 카메라 권한을 요청합니다.

        // 이미지 처리기, 모델, UI 컴포넌트 초기화
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(192, 192, ResizeOp.ResizeMethod.BILINEAR))  // 이미지를 192x192로 리사이즈
            .build()
        model = LiteModelMovenetSingleposeLightningTfliteFloat164.newInstance(this)  // 모델 인스턴스 생성
        imageView = findViewById(R.id.imageView)  // 레이아웃에서 ImageView 찾기
        textureView = findViewById(R.id.textureView)  // 레이아웃에서 TextureView 찾기
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager  // 카메라 서비스 가져오기
        handlerThread = HandlerThread("videoThread")  // 비디오 처리를 위한 별도의 스레드 시작
        handlerThread.start()
        handler = Handler(handlerThread.looper)  // 핸들러에 스레드의 루퍼 연결

        paint.color = Color.YELLOW  // 원을 그릴 때 사용할 색상 설정

        // TextureView가 사용 가능한지 감지하는 리스너 설정
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()  // 텍스처 뷰가 사용 가능하면 카메라를 엽니다.
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
                // 텍스처 뷰의 크기가 변경되었을 때 필요한 처리를 할 수 있습니다.
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                // 텍스처 뷰가 파괴될 때 호출됩니다. 여기서는 특별한 처리가 필요 없습니다.
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                // 텍스처 뷰가 업데이트될 때 마다 실행되는 코드입니다.
                handler.post {
                    bitmap = textureView.bitmap!!  // 텍스처 뷰에서 비트맵을 가져옵니다.
                    var tensorImage = TensorImage(DataType.UINT8)
                    tensorImage.load(bitmap)  // 비트맵으로 TensorImage를 로드
                    tensorImage = imageProcessor.process(tensorImage)  // 이미지 처리

                    // 모델 입력 준비
                    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.UINT8)
                    inputFeature0.loadBuffer(tensorImage.buffer)  // 처리된 이미지를 텐서 버퍼에 로드

                    val outputs = model.process(inputFeature0)  // 모델 실행
                    val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray  // 결과 추출

                    runOnUiThread {
                        var mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                        val canvas = Canvas(mutableBitmap)
                        val h = bitmap.height
                        val w = bitmap.width
                        var x = 0  // 키포인트 인덱스 초기화

                        // 키포인트 그리기
                        Log.d("output__", outputFeature0.size.toString())
                        while (x <= 49) {
                            if (outputFeature0[x + 2] > 0.45) {  // 신뢰도 검사
                                canvas.drawCircle(outputFeature0[x + 1] * w, outputFeature0[x] * h, 10f, paint)
                            }
                            x += 3  // 다음 키포인트로 이동
                        }

                        imageView.setImageBitmap(mutableBitmap)  // 이미지 뷰에 그린 비트맵 설정
                    }
                }
            }
        }
    }

    // 앱이 종료될 때 호출되는 메서드입니다.
    override fun onDestroy() {
        super.onDestroy()
        model.close()  // 사용이 끝난 모델을 종료합니다.
    }

    // 카메라를 열어 화면에 보여주는 메서드입니다.
    @SuppressLint("MissingPermission")
    private fun open_camera() {
        cameraManager.openCamera(cameraManager.cameraIdList[0], object : CameraDevice.StateCallback() {
            override fun onOpened(p0: CameraDevice) {
                var captureRequest = p0.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                var surface = Surface(textureView.surfaceTexture)
                captureRequest.addTarget(surface)
                p0.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)  // 프리뷰 요청을 반복 설정
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                        // 구성 실패 시 처리를 추가할 수 있습니다.
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {
                // 카메라 연결이 끊긴 경우 처리를 추가할 수 있습니다.
            }

            override fun onError(p0: CameraDevice, p1: Int) {
                // 카메라 오류 발생 시 처리를 추가할 수 있습니다.
            }
        }, handler)
    }

    // 카메라 권한을 요청하는 메서드입니다.
    fun get_permissions() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)  // 권한 요청
        }
    }

    // 권한 요청 결과를 처리하는 메서드입니다.
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) get_permissions()  // 권한 재요청
    }
}
