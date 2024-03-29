package com.example.mindsporefederatedlearning;

import static android.content.ContentValues.TAG;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.example.mindsporefederatedlearning.utils.LoggerUtil;
import com.mindspore.Graph;
import com.mindspore.Model;
import com.mindspore.config.DeviceType;
import com.mindspore.config.MSContext;
import com.mindspore.config.TrainCfg;
import com.mindspore.flclient.FLClientStatus;

import com.mindspore.lite.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;


@RequiresApi(api = Build.VERSION_CODES.P)
public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private Button show_log_info;
    private TextView tv_log;
    private String parentPath;

    private FlJob flJob;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        // 获取该应用程序在Android系统中的磁盘路径
        this.parentPath = this.getExternalFilesDir(null).getAbsolutePath();
        // copy assets目录下面的资源文件到Android系统的磁盘中
        AssetCopyer.copyAllAssets(this.getApplicationContext(), parentPath);

        // 初始化日志
        String logFolderPath = parentPath + "/log";
        File logFolder = new File(logFolderPath);
        if(!logFolder.exists()){
            logFolder.mkdir();
        }
        // copy assets目录下面的资源文件到Android系统的磁盘中
        AssetCopyer.copyAllAssets(this.getApplicationContext(), parentPath);
        LoggerUtil.setLogFilePath(parentPath + "/log/MyLogFile.log");

        // 新建一个线程，启动联邦学习训练与推理任务
        Button start = (Button) findViewById(R.id.start_federated_learning);
        start.setOnClickListener(this);
        show_log_info = (Button) findViewById(R.id.bt_show_log_info);
        show_log_info.setOnClickListener(this);
        tv_log = (TextView) findViewById(R.id.tv_debug_info);
        tv_log.setOnClickListener(this);

        Button load_model = (Button) findViewById(R.id.bt_load_model);
        load_model.setOnClickListener(this);
    }

    public String readDataFile(String fileName) {
        String res = "";
        try {
            File file = new File(fileName);
            FileInputStream fin = new FileInputStream(file);
            int length = fin.available();
            byte[] buffer = new byte[length];
            fin.read(buffer);
            res = new String(buffer);
            fin.close();

        } catch (Exception e) {
            e.printStackTrace();
            Log.e("Exception", "readDataFile Error!" + e.getMessage());
        }
        return res;
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.start_federated_learning:
                new Thread(() -> {
                    while(true) {
                        flJob = new FlJob(parentPath);
                        FLClientStatus result = flJob.syncJobTrain();
                        flJob.syncJobPredict();
                        flJob.finish_job();
                        if (result == FLClientStatus.FAILED) {
                            Log.d("FLClientStatus", "FAILED");
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(MainActivity.this, "训练失败，已经自动重启。", Toast.LENGTH_SHORT).show();
                                }
                            });
                        }else {
                            break;
                        }
                    }
                }).start();
                break;
            case R.id.bt_show_log_info:
                String log = readDataFile(parentPath + "/log/MyLogFile.log");
                tv_log.setText(log);
                break;
            case R.id.bt_load_model:
                test_ms();
                break;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        flJob.finish_job();
    }

    public void test_ms() {


// 打印版本

        MSContext context = new MSContext();
        String modelPath = this.parentPath  + "/model/MEAN_MLP_qyj_train.ms";
        Log.d(TAG,"modelPath "+modelPath);
        // use default param init context
        context.init(1, 0);
        boolean isSuccess = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);

        Log.d(TAG,"isSuccess "+isSuccess);
        TrainCfg trainCfg = new TrainCfg();
        trainCfg.init();
        Model model = new Model();
        Graph graph = new Graph();
        boolean g = graph.load(modelPath);
        Log.d(TAG,"xxxxxxxxxxload "+g);
        boolean r = model.build(graph, context, trainCfg);
        Log.d(TAG,"xxxxxxxxxxxxxxbuild "+r);
//        model.setupVirtualBatch(virtualBatch, 0.01f, 1.00f);

    }

//    public void test_ms(){
//        Log.d("qiyijie", "SUCCESS");
//        Model model = new Model();
//        // Create and init config.
//        MSContext context = new MSContext();
//        if (!context.init(2, CpuBindMode.MID_CPU, false)) {
//            Log.e(TAG, "Init context failed");
//        }
//        if (!context.addDeviceInfo(DeviceType.DT_CPU, false, 0)) {
//            Log.e(TAG, "Add device info failed");
//        }
//
//        MappedByteBuffer modelBuffer = loadModel(getApplicationContext(), "MEAN_MLP_qyj_train.ms");
//        if(modelBuffer == null) {
//            Log.e(TAG, "Load model failed");
//        }
//        // build model.
////        boolean ret = model.build(modelBuffer, ModelType.MT_MINDIR,context);
//        String p = this.parentPath  + "/model/albert_inference.mindir.ms";
//        Log.d("TAG","p "+p);
//        boolean ret = model.build(p, ModelType.MT_MINDIR,context);
//        if(!ret) {
//            Log.e(TAG, "Build model failed");
//        }
//
//    }

    private MappedByteBuffer loadModel(Context context, String modelName) {
        FileInputStream fis = null;
        AssetFileDescriptor fileDescriptor = null;

        try {
            fileDescriptor = context.getAssets().openFd(modelName);
            fis = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = fis.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLen = fileDescriptor.getDeclaredLength();
            Log.d("qiyijie", "Loading SUCCESS");
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLen);
        } catch (IOException var24) {
            var24.printStackTrace();
            Log.e("MS_LITE", "Load model failed");
        } finally {
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException var23) {
                    Log.e("MS_LITE", "Close file failed");
                }
            }

            if (fileDescriptor != null) {
                try {
                    fileDescriptor.close();
                } catch (IOException var22) {
                    Log.e("MS_LITE", "Close fileDescriptor failed");
                }
            }

        }

        return null;
    }

}
