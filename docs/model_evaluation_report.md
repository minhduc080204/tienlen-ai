# Báo cáo Ý nghĩa Các Thông số Đánh giá Model (Tiến Lên AI)

Sau khi quá trình huấn luyện và đánh giá (evaluation) hoàn tất, các thông số được lưu trữ trong thư mục `logs/` dưới định dạng CSV. Dưới đây là giải thích chi tiết về ý nghĩa của từng chỉ số và cách diễn giải chúng để đánh giá sức mạnh của AI.

## 1. Các Chỉ số Cơ bản (Core Metrics)

### 1.1. Win Rate (Tỉ lệ thắng)
- **Ý nghĩa:** Tỉ lệ phần trăm số ván AI thắng (về nhất) trên tổng số ván chơi trong đợt đánh giá hoặc một khoảng thời gian huấn luyện.
- **Cách diễn giải:**
    - **Tăng dần:** Model đang học được cách tối ưu hóa việc đánh bài để thắng.
    - **> 25% (với 4 người chơi):** AI đang chơi tốt hơn mức ngẫu nhiên.
    - **> 50%:** AI đang vượt trội so với các đối thủ hiện tại (ví dụ: RuleBots).

### 1.2. Average Reward (Phần thưởng trung bình)
- **Ý nghĩa:** Giá trị trung bình của tổng phần thưởng mà AI nhận được trong một ván đấu. Phần thưởng thường bao gồm điểm cộng khi thắng, điểm trừ khi thua, hoặc điểm thưởng cho các hành động đặc biệt (chặt heo, tứ quý...).
- **Cách diễn giải:**
    - Chỉ số này phản ánh "chất lượng" chiến thắng hoặc thất bại. Ví dụ: Thắng "tới trắng" sẽ có reward cao hơn thắng bình thường. Thua mà còn ít bài sẽ bị trừ ít điểm hơn thua "bét".

### 1.3. Average Turns (Số lượt đánh trung bình)
- **Ý nghĩa:** Số lượt (turns) trung bình để kết thúc một ván đấu.
- **Cách diễn giải:**
    - **Giảm dần:** AI học được cách đánh bài hiệu quả hơn, kết thúc ván đấu nhanh hơn.
    - **Quá cao:** Có thể AI đang chơi quá an toàn hoặc chưa biết cách dứt điểm ván đấu.

### 1.4. Average Entropy (Độ hỗn loạn/Sự thăm dò)
- **Ý nghĩa:** Đo lường mức độ "phân vân" của AI khi chọn nước đi. Entropy cao nghĩa là AI đang thử nghiệm nhiều nước đi khác nhau (Exploration). Entropy thấp nghĩa là AI đang tự tin vào một nước đi cụ thể (Exploitation).
- **Cách diễn giải:**
    - **Giảm dần theo thời gian:** Là dấu hiệu tốt, cho thấy model đang hội tụ (converge) và trở nên tự tin hơn.
    - **Giảm quá nhanh về 0:** Có thể model bị "quá tải" (overfitting) hoặc rơi vào cực tiểu cục bộ quá sớm.

---

## 2. Thống kê Loại nước đi (Move Type Stats)

Các chỉ số này cho biết tần suất AI sử dụng các kiểu kết hợp bài khác nhau:

- **move_single:** Đánh bài rác (1 lá). Tần suất quá cao có thể do AI chưa biết cách ghép bộ.
- **move_pair / move_triple:** Đánh đôi, ba. Cho thấy khả năng quản lý bài bộ.
- **move_straight:** Đánh sảnh. Một trong những kỹ năng quan trọng nhất trong Tiến Lên.
- **move_four_of_kind / move_double_straight:** Tứ quý, đôi thông. Đây là các bộ bài mạnh dùng để "chặt". Nếu các chỉ số này tăng lên, AI đã học được cách giữ và sử dụng các bộ bài quyền lực.
- **move_two:** Đánh heo (2). AI học được khi nào nên tung "át chủ bài".
- **move_pass:** Bỏ lượt. AI học được cách nhường lượt để dành bài cho các vòng sau.

---

## 3. Cách Đọc File Logs

Hệ thống sinh ra hai loại file trong thư mục `logs/`:

1.  **`metrics_<timestamp>.csv` (Training Metrics):**
    - Ghi lại hiệu suất của AI **trong lúc đang học**.
    - Chỉ số này có thể biến động mạnh vì AI vừa chơi vừa thăm dò (exploration).
2.  **`eval_metrics_<timestamp>.csv` (Evaluation Metrics):**
    - Ghi lại hiệu suất khi AI đấu thử nghiệm với RuleBots (thường dùng chế độ `greedy` - chọn nước đi tốt nhất mà không thăm dò).
    - Đây là chỉ số **khách quan nhất** để đánh giá thực lực thực sự của model.

## 4. Dấu hiệu một Model "Tốt"
- **Win Rate** trong Eval đạt trên 40-50% khi đấu với RuleBot.
- **Average Reward** tăng ổn định và đạt giá trị dương.
- **Entropy** giảm dần nhưng không chạm 0 quá nhanh.
- **Move Type Stats** đa dạng, không chỉ tập trung vào đánh bài rác (`single`).
