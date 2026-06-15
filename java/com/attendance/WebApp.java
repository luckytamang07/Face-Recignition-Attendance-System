package com.attendance;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.sql.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@SpringBootApplication
@Controller
public class WebApp {
    
    private static final String BASE_DIR = System.getProperty("user.dir");
    private static final String MODELS_DIR = BASE_DIR + "/src/main/resources/models";
    private static final String YUNET_MODEL = MODELS_DIR + "/face_detection_yunet_2023mar.onnx";
    private static final String SFACE_MODEL = MODELS_DIR + "/face_recognition_sface_2021dec.onnx";
    
    // Modern detectors
    private static FaceDetectorYN faceDetector;        // YuNet - 95% accuracy
    private static FaceRecognizerSF faceRecognizer;    // SFace - 99% accuracy
    
    // Store face embeddings
    private static Map<String, List<float[]>> faceEmbeddings = new ConcurrentHashMap<>();
    
    static {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();
        
        try {
            // Load YuNet face detector using FaceDetectorYN
            faceDetector = FaceDetectorYN.create(YUNET_MODEL, "", new Size(320, 320));
            if (faceDetector == null) {
                System.err.println("❌ YuNet model not loaded - check file exists");
            } else {
                System.out.println("✅ YuNet face detector loaded (95% accuracy)");
            }
            
            // Load SFace face recognizer
            faceRecognizer = FaceRecognizerSF.create(SFACE_MODEL, "");
            if (faceRecognizer == null) {
                System.err.println("❌ SFace model not loaded");
            } else {
                System.out.println("✅ SFace face recognizer loaded (99% accuracy)");
            }
            
        } catch (Exception e) {
            System.err.println("❌ Model loading error: " + e.getMessage());
            System.err.println("Make sure models are downloaded to: " + MODELS_DIR);
        }
    }
    
    public static void main(String[] args) {
        SpringApplication.run(WebApp.class, args);
        initDatabase();
        createFacesDirectory();
        loadFaceEmbeddings();
        System.out.println("\n╔══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║     MODERN FACE RECOGNITION ATTENDANCE SYSTEM                        ║");
        System.out.println("║     Detector: YuNet (95% accuracy)                                   ║");
        System.out.println("║     Recognizer: SFace (99% accuracy)                                 ║");
        System.out.println("║     Flow: Webcam → YuNet Detection → SFace Embedding → Cosine → Mark ║");
        System.out.println("║     Open: http://localhost:8080                                      ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════╝\n");
    }
    
    private static void createFacesDirectory() {
        File facesDir = new File(BASE_DIR + "/faces");
        if (!facesDir.exists()) {
            facesDir.mkdirs();
        }
    }
    
    private static void loadFaceEmbeddings() {
        String sql = "SELECT name, face_embedding FROM users WHERE face_embedding IS NOT NULL";
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                String name = rs.getString("name");
                byte[] bytes = rs.getBytes("face_embedding");
                if (bytes != null) {
                    float[] embedding = new float[bytes.length / 4];
                    for (int i = 0; i < embedding.length; i++) {
                        embedding[i] = java.nio.ByteBuffer.wrap(bytes, i * 4, 4).getFloat();
                    }
                    faceEmbeddings.computeIfAbsent(name, k -> new ArrayList<>()).add(embedding);
                }
            }
            System.out.println("✅ Loaded " + faceEmbeddings.size() + " face embeddings");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    @GetMapping("/")
    public String home() {
        return "index";
    }
    
    @GetMapping("/register")
    public String registerPage() {
        return "register";
    }
    
    @GetMapping("/mark")
    public String markPage() {
        return "mark";
    }
    
    @GetMapping("/report")
    public String reportPage(Model model) throws Exception {
        model.addAttribute("records", getAttendanceRecords());
        return "report";
    }
    
    @GetMapping("/api/people")
    @ResponseBody
    public List<String> getPeople() {
        return new ArrayList<>(faceEmbeddings.keySet());
    }
    
    // ==================== STEP 1: MODERN FACE DETECTION (YuNet using FaceDetectorYN) ====================
    private Mat detectFaceYuNet(Mat image) {
        try {
            if (faceDetector == null) {
                System.err.println("Face detector not initialized");
                return null;
            }
            
            // Set input size to match the image
            faceDetector.setInputSize(image.size());
            
            // Detect faces - returns Mat with shape [num_faces, 15]
            Mat faces = new Mat();
            int detectionCount = faceDetector.detect(image, faces);
            
            if (detectionCount > 0 && faces.rows() > 0) {
                // Return the first detection row (contains bounding box + landmarks)
                return faces.row(0);
            }
        } catch (Exception e) {
            System.err.println("YuNet detection error: " + e.getMessage());
        }
        return null;
    }
    
    // Helper to extract face crop from detection Mat (for saving face images)
    private Mat extractFaceCropFromDetection(Mat image, Mat detection) {
        try {
            int cols = image.cols();
            int rows = image.rows();
            
            // Detection format: [x, y, width, height, ...]
            float x1 = (float) detection.get(0, 0)[0] * cols;
            float y1 = (float) detection.get(0, 1)[0] * rows;
            float w = (float) detection.get(0, 2)[0] * cols;
            float h = (float) detection.get(0, 3)[0] * rows;
            
            int left = Math.max(0, (int) x1);
            int top = Math.max(0, (int) y1);
            int right = Math.min(cols - 1, (int) (x1 + w));
            int bottom = Math.min(rows - 1, (int) (y1 + h));
            
            if (right > left && bottom > top) {
                return new Mat(image, new Rect(left, top, right - left, bottom - top));
            }
        } catch (Exception e) {
            System.err.println("Crop extraction error: " + e.getMessage());
        }
        return null;
    }
    
    // ==================== STEP 2: EXTRACT SFACE EMBEDDING ====================
    private float[] extractFaceEmbeddingSFace(Mat detection, Mat originalImage) {
        try {
            if (faceRecognizer == null) {
                System.err.println("Face recognizer not initialized");
                return null;
            }
            
            // alignCrop expects: (src_img, face_box, aligned_img)
            // The detection Mat from FaceDetectorYN contains the face bounding box in the correct format
            Mat alignedFace = new Mat();
            faceRecognizer.alignCrop(originalImage, detection, alignedFace);
            
            // Extract features (512-dimensional embedding)
            Mat embeddingMat = new Mat();
            faceRecognizer.feature(alignedFace, embeddingMat);
            
            float[] embedding = new float[embeddingMat.cols()];
            embeddingMat.get(0, 0, embedding);
            
            return embedding;
        } catch (Exception e) {
            System.err.println("SFace embedding error: " + e.getMessage());
            return null;
        }
    }
    
    // ==================== STEP 3: COSINE SIMILARITY ====================
    private double cosineSimilarity(float[] emb1, float[] emb2) {
        if (emb1.length != emb2.length) return 0;
        
        double dotProduct = 0;
        double norm1 = 0;
        double norm2 = 0;
        
        for (int i = 0; i < emb1.length; i++) {
            dotProduct += emb1[i] * emb2[i];
            norm1 += emb1[i] * emb1[i];
            norm2 += emb2[i] * emb2[i];
        }
        
        if (norm1 == 0 || norm2 == 0) return 0;
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    private double similarityPercentage(double cosineSim) {
        // Cosine similarity ranges from -1 to 1, convert to 0-100
        return (cosineSim + 1) / 2 * 100;
    }
    
    // ==================== API: REGISTER FACE ====================
    @PostMapping("/api/register")
    @ResponseBody
    public Map<String, Object> registerFace(@RequestParam String name, @RequestParam MultipartFile photo) {
        Map<String, Object> response = new HashMap<>();
        try {
            // Check duplicate name
            if (faceEmbeddings.containsKey(name)) {
                response.put("success", false);
                response.put("message", "❌ Name '" + name + "' already registered!");
                return response;
            }
            
            // Convert MultipartFile to OpenCV Mat
            BufferedImage bufferedImage = ImageIO.read(photo.getInputStream());
            Mat image = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CvType.CV_8UC3);
            byte[] data = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
            image.put(0, 0, data);
            Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2BGR);
            
            // STEP 1: Face Detection (YuNet)
            Mat detectionResult = detectFaceYuNet(image);
            if (detectionResult == null) {
                response.put("success", false);
                response.put("message", "❌ No face detected! Please ensure good lighting and look straight at camera.");
                return response;
            }
            
            // Extract face crop for saving (optional)
            Mat faceCrop = extractFaceCropFromDetection(image, detectionResult);
            
            // STEP 2: Extract SFace Embedding
            float[] embedding = extractFaceEmbeddingSFace(detectionResult, image);
            if (embedding == null) {
                response.put("success", false);
                response.put("message", "❌ Failed to extract face features!");
                return response;
            }
            
            // Check for duplicate face (prevent same person registering twice)
            for (Map.Entry<String, List<float[]>> entry : faceEmbeddings.entrySet()) {
                for (float[] existingEmbedding : entry.getValue()) {
                    double similarity = cosineSimilarity(embedding, existingEmbedding);
                    double percentage = similarityPercentage(similarity);
                    if (percentage > 75) {  // 75% threshold for duplicate face
                        response.put("success", false);
                        response.put("message", "❌ This face already registered as '" + entry.getKey() + "'! (Match: " + Math.round(percentage) + "%)");
                        return response;
                    }
                }
            }
            
            // Save face image (optional - for reference)
            if (faceCrop != null) {
                String filename = BASE_DIR + "/faces/" + name + "_" + System.currentTimeMillis() + ".png";
                Imgcodecs.imwrite(filename, faceCrop);
            }
            
            // Save embedding to database
            byte[] embeddingBytes = new byte[embedding.length * 4];
            for (int i = 0; i < embedding.length; i++) {
                java.nio.ByteBuffer.wrap(embeddingBytes, i * 4, 4).putFloat(embedding[i]);
            }
            
            String sql = "INSERT INTO users (name, face_embedding, registered_date) VALUES (?, ?, ?)";
            try (Connection conn = getConnection();
                 PreparedStatement pstmt = conn.prepareStatement(sql)) {
                pstmt.setString(1, name);
                pstmt.setBytes(2, embeddingBytes);
                pstmt.setString(3, LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
                pstmt.executeUpdate();
            }
            
            faceEmbeddings.computeIfAbsent(name, k -> new ArrayList<>()).add(embedding);
            
            response.put("success", true);
            response.put("message", "✅ Face registered for " + name + "!");
            
        } catch (Exception e) {
            response.put("success", false);
            response.put("message", "❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
        return response;
    }
    
    // ==================== API: RECOGNIZE FACE ====================
    @PostMapping("/api/recognize")
    @ResponseBody
    public Map<String, Object> recognizeFace(@RequestParam MultipartFile photo) {
        Map<String, Object> response = new HashMap<>();
        try {
            if (faceEmbeddings.isEmpty()) {
                response.put("success", false);
                response.put("message", "❌ No registered faces! Please register first.");
                return response;
            }
            
            // Convert MultipartFile to OpenCV Mat
            BufferedImage bufferedImage = ImageIO.read(photo.getInputStream());
            Mat image = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CvType.CV_8UC3);
            byte[] data = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
            image.put(0, 0, data);
            Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2BGR);
            
            // STEP 1: Face Detection (YuNet)
            Mat detectionResult = detectFaceYuNet(image);
            if (detectionResult == null) {
                response.put("success", false);
                response.put("message", "❌ No face detected! Please look at camera.");
                return response;
            }
            
            // STEP 2: Extract SFace Embedding
            float[] capturedEmbedding = extractFaceEmbeddingSFace(detectionResult, image);
            if (capturedEmbedding == null) {
                response.put("success", false);
                response.put("message", "❌ Failed to extract face features!");
                return response;
            }
            
            // STEP 3: Cosine Similarity Matching
            String bestMatch = null;
            double bestSimilarity = -1;
            double bestPercentage = 0;
            
            for (Map.Entry<String, List<float[]>> entry : faceEmbeddings.entrySet()) {
                for (float[] storedEmbedding : entry.getValue()) {
                    double similarity = cosineSimilarity(capturedEmbedding, storedEmbedding);
                    double percentage = similarityPercentage(similarity);
                    
                    if (similarity > bestSimilarity && percentage > 65) {  // 65% threshold for recognition
                        bestSimilarity = similarity;
                        bestPercentage = percentage;
                        bestMatch = entry.getKey();
                    }
                }
            }
            
            if (bestMatch != null) {
                // Check duplicate attendance
                String checkSql = "SELECT COUNT(*) FROM attendance WHERE user_name = ? AND date = date('now')";
                try (Connection conn = getConnection();
                     PreparedStatement pstmt = conn.prepareStatement(checkSql)) {
                    pstmt.setString(1, bestMatch);
                    ResultSet rs = pstmt.executeQuery();
                    rs.next();
                    if (rs.getInt(1) > 0) {
                        response.put("success", false);
                        response.put("message", "⚠️ " + bestMatch + ", already marked today! (Match: " + Math.round(bestPercentage) + "%)");
                        response.put("name", bestMatch);
                        return response;
                    }
                }
                
                // Mark attendance
                String sql = "INSERT INTO attendance (user_name, date, time) VALUES (?, date('now'), time('now'))";
                try (Connection conn = getConnection();
                     PreparedStatement pstmt = conn.prepareStatement(sql)) {
                    pstmt.setString(1, bestMatch);
                    pstmt.executeUpdate();
                }
                
                response.put("success", true);
                response.put("message", "✅ Welcome " + bestMatch + "! (Match: " + Math.round(bestPercentage) + "%)");
                response.put("name", bestMatch);
                response.put("similarity", bestSimilarity);
                response.put("confidence", bestPercentage);
            } else {
                response.put("success", false);
                response.put("message", "❌ Face not recognized! Please register first.");
            }
        } catch (Exception e) {
            response.put("success", false);
            response.put("message", "❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
        return response;
    }
    
    // ==================== API: GET ATTENDANCE REPORT ====================
    @GetMapping("/api/report")
    @ResponseBody
    public List<Map<String, String>> getReport() throws Exception {
        List<Map<String, String>> records = new ArrayList<>();
        String sql = "SELECT id, user_name, date, time FROM attendance ORDER BY date DESC, time DESC";
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                Map<String, String> record = new HashMap<>();
                record.put("id", String.valueOf(rs.getInt("id")));
                record.put("name", rs.getString("user_name"));
                record.put("date", rs.getString("date"));
                record.put("time", rs.getString("time"));
                records.add(record);
            }
        }
        return records;
    }
    
    // ==================== API: DELETE ATTENDANCE RECORD ====================
    @DeleteMapping("/api/attendance/{id}")
    @ResponseBody
    public Map<String, Object> deleteAttendance(@PathVariable int id) {
        Map<String, Object> response = new HashMap<>();
        try {
            String sql = "DELETE FROM attendance WHERE id = ?";
            try (Connection conn = getConnection();
                 PreparedStatement pstmt = conn.prepareStatement(sql)) {
                pstmt.setInt(1, id);
                int deleted = pstmt.executeUpdate();
                response.put("success", deleted > 0);
                response.put("message", deleted > 0 ? "Deleted" : "Not found");
            }
        } catch (Exception e) {
            response.put("success", false);
            response.put("message", e.getMessage());
        }
        return response;
    }
    
    // ==================== DATABASE HELPERS ====================
    private static Connection getConnection() throws SQLException {
        String dbPath = "jdbc:sqlite:" + BASE_DIR + "/attendance.db";
        return DriverManager.getConnection(dbPath);
    }
    
    private List<Map<String, String>> getAttendanceRecords() throws SQLException {
        List<Map<String, String>> records = new ArrayList<>();
        String sql = "SELECT id, user_name, date, time FROM attendance ORDER BY date DESC, time DESC";
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                Map<String, String> record = new HashMap<>();
                record.put("id", String.valueOf(rs.getInt("id")));
                record.put("name", rs.getString("user_name"));
                record.put("date", rs.getString("date"));
                record.put("time", rs.getString("time"));
                records.add(record);
            }
        }
        return records;
    }
    
    static void initDatabase() {
        String userTable = "CREATE TABLE IF NOT EXISTS users (" +
                          "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                          "name TEXT NOT NULL UNIQUE," +
                          "face_embedding BLOB," +
                          "registered_date TEXT)";
        
        String attendanceTable = "CREATE TABLE IF NOT EXISTS attendance (" +
                                "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                                "user_name TEXT NOT NULL," +
                                "date TEXT NOT NULL," +
                                "time TEXT NOT NULL," +
                                "UNIQUE(user_name, date))";
        
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement()) {
            stmt.execute(userTable);
            stmt.execute(attendanceTable);
            System.out.println("✅ Database initialized");
        } catch (SQLException e) {
            System.out.println("Database error: " + e.getMessage());
        }
    }
}