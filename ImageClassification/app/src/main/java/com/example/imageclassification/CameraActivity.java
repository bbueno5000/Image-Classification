package com.example.imageclassification;

/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;
import com.example.imageclassification.env.ImageUtils;
import com.example.imageclassification.env.Logger;
import com.example.tflite.Classifier;
import com.example.tflite.Classifier.Device;
import com.example.tflite.Classifier.Recognition;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import java.nio.ByteBuffer;
import java.util.List;

/**
 *
 */
public abstract class CameraActivity extends AppCompatActivity implements OnImageAvailableListener,
        Camera.PreviewCallback, View.OnClickListener, AdapterView.OnItemSelectedListener {

    protected ImageView bottomSheetArrowImageView;
    private LinearLayout bottomSheetLayout;
    protected TextView cameraResolutionTextView;
    protected TextView cropValueTextView;
    private Device device = Device.CPU;
    private Spinner deviceSpinner;
    protected TextView frameValueTextView;
    private LinearLayout gestureLayout;
    private Handler handler;
    private HandlerThread handlerThread;
    private Runnable imageConverter;
    protected TextView inferenceTimeTextView;
    private boolean isProcessingFrame = false;
    private static final Logger LOGGER = new Logger();
    private ImageView minusImageView;
    private Classifier.Model model = Classifier.Model.QUANTIZED_EFFICIENTNET;
    private int numThreads = -1;
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private ImageView plusImageView;
    private Runnable postInferenceCallback;
    protected int previewWidth = 0;
    protected int previewHeight = 0;
    protected TextView recognitionTextView;
    protected TextView recognition1TextView;
    protected TextView recognition2TextView;
    protected TextView recognitionValueTextView;
    protected TextView recognition1ValueTextView;
    protected TextView recognition2ValueTextView;
    private int[] rgbBytes = null;
    protected TextView rotationTextView;
    private BottomSheetBehavior<LinearLayout> sheetBehavior;
    private TextView threadsTextView;
    private boolean useCamera2API;
    private int yRowStride;
    private byte[][] yuvBytes = new byte[3][];

    /**
     *
     * @param grantResults
     * @return
     */
    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    /**
     *
     * @return
     */
    private String chooseCamera() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
                // We don't use a front facing camera in this sample.
                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }
                final StreamConfigurationMap map = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map == null) {
                    continue;
                }
                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API = (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                        || isHardwareLevelSupported(characteristics,
                        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
                LOGGER.i("Camera API lv2?: %s", useCamera2API);
                return cameraId;
            }
        } catch (CameraAccessException e) {
            LOGGER.e(e, "Not allowed to access camera");
        }
        return null;
    }

    /**
     *
     * @param planes
     * @param yuvBytes
     */
    protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    /**
     *
     * @return
     */
    protected abstract Size getDesiredPreviewFrameSize();

    /**
     *
     * @return
     */
    protected Device getDevice() {
        return device;
    }

    /**
     *
     * @return
     */
    protected abstract int getLayoutId();

    /**
     *
     * @return
     */
    protected byte[] getLuminance() {
        return yuvBytes[0];
    }

    /**
     *
     * @return
     */
    protected int getLuminanceStride() {
        return yRowStride;
    }

    /**
     *
     * @return
     */
    protected Classifier.Model getModel() {
        return model;
    }

    /**
     *
     * @return
     */
    protected int getNumThreads() {
        return numThreads;
    }

    /**
     *
     * @return
     */
    protected int[] getRgbBytes() {
        imageConverter.run();
        return rgbBytes;
    }

    /**
     *
     * @return
     */
    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    /**
     *
     * @return
     */
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    /**
     *
     * @param characteristics
     * @param requiredLevel
     * @return Returns true if the device supports the required hardware level, or better.
     */
    private boolean isHardwareLevelSupported(
            CameraCharacteristics characteristics, int requiredLevel) {
        int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            return requiredLevel == deviceLevel;
        }
        // deviceLevel is not LEGACY, can use numerical sort
        return requiredLevel <= deviceLevel;
    }

    /**
     *
     * @param v
     */
    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.plus) {
            String threads = threadsTextView.getText().toString().trim();
            int numThreads = Integer.parseInt(threads);
            if (numThreads >= 9) {
                return;
            }
            setNumThreads(++numThreads);
            threadsTextView.setText(String.valueOf(numThreads));
        } else if (v.getId() == R.id.minus) {
            String threads = threadsTextView.getText().toString().trim();
            int numThreads = Integer.parseInt(threads);
            if (numThreads == 1) {
                return;
            }
            setNumThreads(--numThreads);
            threadsTextView.setText(String.valueOf(numThreads));
        }
    }

    /**
     *
     * @param savedInstanceState
     */
    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        LOGGER.d("onCreate " + this);
        super.onCreate(null);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.tfe_ic_activity_camera);
        if (hasPermission()) {
            setFragment();
        } else {
            requestPermission();
        }
        threadsTextView = findViewById(R.id.threads);
        plusImageView = findViewById(R.id.plus);
        minusImageView = findViewById(R.id.minus);
        deviceSpinner = findViewById(R.id.device_spinner);
        bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
        gestureLayout = findViewById(R.id.gesture_layout);
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
        bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);
        ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
        vto.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
                    gestureLayout.getViewTreeObserver().removeGlobalOnLayoutListener(this);
                } else {
                    gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                }
                //int width = bottomSheetLayout.getMeasuredWidth();
                int height = gestureLayout.getMeasuredHeight();
                sheetBehavior.setPeekHeight(height);
            }
        });
        sheetBehavior.setHideable(false);
        sheetBehavior.setBottomSheetCallback(new BottomSheetBehavior.BottomSheetCallback() {
            @Override
            public void onStateChanged(@NonNull View bottomSheet, int newState) {
                switch (newState) {
                    case BottomSheetBehavior.STATE_HIDDEN:
                    case BottomSheetBehavior.STATE_DRAGGING: {
                        break;
                    }
                    case BottomSheetBehavior.STATE_EXPANDED: {
                        bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                        break;
                    }
                    case BottomSheetBehavior.STATE_COLLAPSED:
                    case BottomSheetBehavior.STATE_SETTLING: {
                        bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                        break;
                    }
                }
            }
            @Override
            public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
        });
        recognitionTextView = findViewById(R.id.detected_item);
        recognitionValueTextView = findViewById(R.id.detected_item_value);
        recognition1TextView = findViewById(R.id.detected_item1);
        recognition1ValueTextView = findViewById(R.id.detected_item1_value);
        recognition2TextView = findViewById(R.id.detected_item2);
        recognition2ValueTextView = findViewById(R.id.detected_item2_value);
        frameValueTextView = findViewById(R.id.frame_info);
        cropValueTextView = findViewById(R.id.crop_info);
        cameraResolutionTextView = findViewById(R.id.view_info);
        rotationTextView = findViewById(R.id.rotation_info);
        inferenceTimeTextView = findViewById(R.id.inference_info);
        deviceSpinner.setOnItemSelectedListener(this);
        plusImageView.setOnClickListener(this);
        minusImageView.setOnClickListener(this);
        device = Device.valueOf(deviceSpinner.getSelectedItem().toString());
        numThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
    }

    /**
     *
     */
    @Override
    public synchronized void onDestroy() {
        LOGGER.d("onDestroy " + this);
        super.onDestroy();
    }

    /**
     * Callback for Camera2 API
     */
    @Override
    public void onImageAvailable(final ImageReader reader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }

            isProcessingFrame = true;
            Trace.beginSection("imageAvailable");
            final Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter = new Runnable() {
                @Override
                public void run() {
                    ImageUtils.convertYUV420ToARGB8888(
                            yuvBytes[0], yuvBytes[1], yuvBytes[2], previewWidth, previewHeight,
                            yRowStride, uvRowStride, uvPixelStride, rgbBytes);
                }
            };

            postInferenceCallback = new Runnable() {
                @Override
                public void run() {
                    image.close();
                    isProcessingFrame = false;
                }
            };

            processImage();

        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            Trace.endSection();
            return;
        }

        Trace.endSection();
    }

    /**
     *
     */
    protected abstract void onInferenceConfigurationChanged();

    /**
     *
     * @param parent
     * @param view
     * @param pos
     * @param id
     */
    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
        if (parent == deviceSpinner) {
            setDevice(Device.valueOf(parent.getItemAtPosition(pos).toString()));
        }
    }

    /**
     *
     * @param parent
     */
    @Override
    public void onNothingSelected(AdapterView<?> parent) { }

    /**
     * Callback for android.hardware.Camera API
     */
    @Override
    public void onPreviewFrame(final byte[] bytes, final Camera camera) {
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!");
            return;
        }
        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                Camera.Size previewSize = camera.getParameters().getPreviewSize();
                previewHeight = previewSize.height;
                previewWidth = previewSize.width;
                rgbBytes = new int[previewWidth * previewHeight];
                onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
            }
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            return;
        }
        isProcessingFrame = true;
        yuvBytes[0] = bytes;
        yRowStride = previewWidth;
        imageConverter = new Runnable() {
            @Override
            public void run() {
                ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
            }
        };
        postInferenceCallback = new Runnable() {
            @Override
            public void run() {
                camera.addCallbackBuffer(bytes);
                isProcessingFrame = false;
            }
        };
        processImage();
    }

    /**
     *
     */
    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }
        super.onPause();
    }

    /**
     *
     * @param size
     * @param rotation
     */
    protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

    /**
     *
     * @param requestCode
     * @param permissions
     * @param grantResults
     */
    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment();
            } else {
                requestPermission();
            }
        }
    }

    /**
     *
     */
    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    /**
     *
     */
    @Override
    public synchronized void onStart() {
        LOGGER.d("onStart " + this);
        super.onStart();
    }

    /**
     *
     */
    @Override
    public synchronized void onStop() {
        LOGGER.d("onStop " + this);
        super.onStop();
    }

    /**
     *
     */
    protected abstract void processImage();

    /**
     *
     */
    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }

    /**
     *
     */
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                        CameraActivity.this,
                        "Camera permission is required for this demo",
                        Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
        }
    }

    /**
     *
     * @param r
     */
    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    /**
     *
     * @param device
     */
    private void setDevice(Device device) {
        if (this.device != device) {
            LOGGER.d("Updating  device: " + device);
            this.device = device;
            final boolean threadsEnabled = device == Device.CPU;
            plusImageView.setEnabled(threadsEnabled);
            minusImageView.setEnabled(threadsEnabled);
            threadsTextView.setText(threadsEnabled ? String.valueOf(numThreads) : "N/A");
            onInferenceConfigurationChanged();
        }
    }

    /**
     *
     */
    protected void setFragment() {
        String cameraId = chooseCamera();
        Fragment fragment;
        if (useCamera2API) {
            CameraConnectionFragment camera2Fragment = CameraConnectionFragment.newInstance(
                    new CameraConnectionFragment.ConnectionCallback() {
                        @Override
                        public void onPreviewSizeChosen(final Size size, final int rotation) {
                            previewHeight = size.getHeight();
                            previewWidth = size.getWidth();
                            CameraActivity.this.onPreviewSizeChosen(size, rotation);
                        }}, this, getLayoutId(), getDesiredPreviewFrameSize());
            camera2Fragment.setCamera(cameraId);
            fragment = camera2Fragment;
        } else {
            fragment = new LegacyCameraConnectionFragment(
                    this, getLayoutId(), getDesiredPreviewFrameSize());
        }
        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }

    /**
     *
     * @param numThreads
     */
    private void setNumThreads(int numThreads) {
        if (this.numThreads != numThreads) {
            LOGGER.d("Updating  numThreads: " + numThreads);
            this.numThreads = numThreads;
            onInferenceConfigurationChanged();
        }
    }

    /**
     *
     * @param cameraInfo
     */
    protected void showCameraResolution(String cameraInfo) {
        cameraResolutionTextView.setText(cameraInfo);
    }

    /**
     *
     * @param cropInfo
     */
    protected void showCropInfo(String cropInfo) {
        cropValueTextView.setText(cropInfo);
    }

    /**
     *
     * @param frameInfo
     */
    protected void showFrameInfo(String frameInfo) {
        frameValueTextView.setText(frameInfo);
    }

    /**
     *
     * @param inferenceTime
     */
    protected void showInference(String inferenceTime) {
        inferenceTimeTextView.setText(inferenceTime);
    }

    /**
     *
     * @param results
     */
    @UiThread
    protected void showResultsInBottomSheet(List<Recognition> results) {
        if (results != null && results.size() >= 3) {
            Recognition recognition = results.get(0);
            if (recognition != null) {
                if (recognition.getTitle() != null) {
                    recognitionTextView.setText(recognition.getTitle());
                }
                if (recognition.getConfidence() != null) {
                    recognitionValueTextView.setText(
                            String.format("%.2f", (100 * recognition.getConfidence())) + "%");
                }
            }
            Recognition recognition1 = results.get(1);
            if (recognition1 != null) {
                if (recognition1.getTitle() != null) {
                    recognition1TextView.setText(recognition1.getTitle());
                }
                if (recognition1.getConfidence() != null) {
                    recognition1ValueTextView.setText(
                            String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
                }
            }
            Recognition recognition2 = results.get(2);
            if (recognition2 != null) {
                if (recognition2.getTitle() != null) {
                    recognition2TextView.setText(recognition2.getTitle());
                }
                if (recognition2.getConfidence() != null) {
                    recognition2ValueTextView.setText(
                            String.format("%.2f", (100 * recognition2.getConfidence())) + "%");
                }
            }
        }
    }

    /**
     *
     * @param rotation
     */
    protected void showRotationInfo(String rotation) {
        rotationTextView.setText(rotation);
    }
}
