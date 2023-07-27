package com.example.mindsporefederatedlearning;

import android.annotation.SuppressLint;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.mindspore.flclient.BindMode;
import com.mindspore.flclient.FLClientStatus;
import com.mindspore.flclient.FLParameter;
import com.mindspore.flclient.SyncFLJob;
import com.mindspore.flclient.model.RunType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
public class FlJob {
    private static final Logger LOGGER = Logger.getLogger(FlJob.class.toString());
    private String parentPath;
    private SyncFLJob train_job;
    public FlJob(String parentPath) {
        this.parentPath = parentPath;
    }
    // Android的联邦学习训练任务
    @SuppressLint("NewApi")
    @RequiresApi(api = Build.VERSION_CODES.M)
    public FLClientStatus syncJobTrain() {
        // 构造dataMap
        String trainTxtPath = this.parentPath + "/data/1.txt";
        String evalTxtPath = this.parentPath + "/data/eval.txt";      // 非必须，getModel之后不进行验证可不设置
        String vocabFile = this.parentPath + "/data/vocab.txt";                // 数据预处理的词典文件路径
        String idsFile = this.parentPath + "/data/vocab_map_ids.txt";   // 词典的映射id文件路径
        Map<RunType, List<String>> dataMap = new HashMap<>();
        List<String> trainPath = new ArrayList<>();
        trainPath.add(trainTxtPath);
        trainPath.add(vocabFile);
        trainPath.add(idsFile);
        List<String> evalPath = new ArrayList<>();    // 非必须，getModel之后不进行验证可不设置
        evalPath.add(evalTxtPath);                  // 非必须，getModel之后不进行验证可不设置
        evalPath.add(vocabFile);                  // 非必须，getModel之后不进行验证可不设置
        evalPath.add(idsFile);                  // 非必须，getModel之后不进行验证可不设置
        dataMap.put(RunType.TRAINMODE, trainPath);
        dataMap.put(RunType.EVALMODE, evalPath);      // 非必须，getModel之后不进行验证可不设置

        String flName = "com.example.mindsporefederatedlearning.albert.AlbertClient";                             // AlBertClient.java 包路径
        String trainModelPath = "/model/albert_inference.mindir.ms";                      // 绝对路径
        String inferModelPath = "/model/albert_inference.mindir.ms";                      // 绝对路径, 和trainModelPath保持一致
        String sslProtocol = "TLSv1.2";
        String deployEnv = "android";

        // 端云通信url，请保证Android能够访问到server，否则会出现connection failed
        String domainName = "http://192.168.199.162:9021";
        boolean ifUseElb = true;
//        String domainName = "http://test-kolun-fl.transsion-os.com";
//        boolean ifUseElb = false;
        int serverNum = 1;
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = 16;

        FLParameter flParameter = FLParameter.getInstance();
        flParameter.setFlName(flName);
        flParameter.setDataMap(dataMap);
        flParameter.setTrainModelPath(this.parentPath+trainModelPath);
        flParameter.setInferModelPath(this.parentPath+inferModelPath);
        flParameter.setSslProtocol(sslProtocol);
        flParameter.setDeployEnv(deployEnv);
        flParameter.setDomainName(domainName);
        flParameter.setUseElb(ifUseElb);
        flParameter.setServerNum(serverNum);
        flParameter.setThreadNum(threadNum);
        flParameter.setCpuBindMode(cpuBindMode);
        flParameter.setBatchSize(batchSize);
        flParameter.setSleepTime(100);

//        List<String> list = new ArrayList<>();
//        list.add(trainTxtPath);
//        List<String> list2 = new ArrayList<>();
//        list2.add(evalTxtPath);
//        flParameter.setHybridWeightName(list, RunType.TRAINMODE);
//        flParameter.setHybridWeightName(list2, RunType.INFERMODE);
        // start FLJob
        train_job = new SyncFLJob();
        return train_job.flJobRun();
    }
    // Android的联邦学习推理任务
    public void syncJobPredict() {
        // 构造dataMap
        String inferTxtPath = this.parentPath + "/data/eval.txt";
        String vocabFile = this.parentPath + "/data/vocab.txt";
        String idsFile = this.parentPath + "/data/vocab_map_ids.txt";
        Map<RunType, List<String>> dataMap = new HashMap<>();
        List<String> inferPath = new ArrayList<>();
        inferPath.add(inferTxtPath);
        inferPath.add(vocabFile);
        inferPath.add(idsFile);
        dataMap.put(RunType.INFERMODE, inferPath);

        String flName = "com.example.mindsporefederatedlearning.albert.AlbertClient";                           // AlBertClient.java 包路径
        String inferModelPath = "/model/albert_supervise.mindir.ms";                      // 绝对路径, 和trainModelPath保持一致;
        int threadNum = 1;
        BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
        int batchSize = 16;

        FLParameter flParameter = FLParameter.getInstance();
        flParameter.setFlName(flName);
        flParameter.setDataMap(dataMap);
        flParameter.setInferModelPath(this.parentPath+inferModelPath);
        flParameter.setThreadNum(threadNum);
        flParameter.setCpuBindMode(cpuBindMode);
        flParameter.setBatchSize(batchSize);

        // inference
        SyncFLJob syncFLJob = new SyncFLJob();
        List<Object> labels = syncFLJob.modelInfer();
        LOGGER.info("labels = " + Arrays.toString(labels.toArray()));
    }

    public void finish_job(){
        train_job.stopFLJob();
    }
}