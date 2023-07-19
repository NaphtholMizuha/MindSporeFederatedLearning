package com.example.mindsporefederatedlearning;

import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.example.mindsporefederatedlearning.utils.LoggerUtil;

import java.io.File;
import java.io.FileInputStream;


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
                    flJob = new FlJob(parentPath);
                    flJob.syncJobTrain();
                    flJob.syncJobPredict();
                }).start();
                break;
            case R.id.bt_show_log_info:
                String log = readDataFile(parentPath + "/log/MyLogFile.log");
                tv_log.setText(log);
                break;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        flJob.finish_job();
    }
}
