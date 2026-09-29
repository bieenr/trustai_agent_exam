# Experiment note: `score_candidate`

Gom toàn bộ các lần đánh giá `score_candidate` (luồng quy tắc cũ/mới, logreg, GBM, kNN, MLP) vào một chỗ.
Cập nhật: 2026-09-28.

## Kết luận

- **Luồng quy tắc (B)** (nay chỉ là fallback khi không có model / user lạ) tốt hơn rõ so với luồng cũ: pass/fail đúng 72.8% so với 51.6% (trên 2000 cặp). Nhưng
  nhánh `genre_deviation` gần như vô dụng (fail đúng 31.8%) và 64.5% cặp vẫn không có verdict.
- **GBM (13 đặc trưng)** là tốt nhất trên toàn bộ tập test: AUC 0.789 / 0.828 (thích / không thích), precision tổng
  82.4% ở độ phủ 23.0%. Nó mạnh nhất ở chiều "không thích".
- **MLP** (học trên 59k rating đã che phim mục tiêu) dự đoán rating tốt hơn baseline bias (test RMSE 0.868 so với 0.932)
  và cho độ phủ cao nhất (31.2%, precision 77.8%). Tuy vậy AUC chỉ ngang logreg (0.774 / 0.807), vẫn thua GBM.
  Thay đổi số chiều (PCA 32–512, user 16–256) hay thêm one-hot genre đều chỉ tạo chênh lệch cỡ nhiễu giữa các seed.
- **Nhãn tương đối** (rating − user_mean ≥ +0.5 / ≤ −0.5) sửa được việc user khó tính gần như không có pass, nhưng
  khó dự đoán hơn nhiều (AUC thích 0.737 so với 0.786) và precision pass chỉ khoảng 63–71% (xem [mục 8](#8-nhãn-tương-đối-theo-trung-bình-user-scriptseval_relative_labels)).
  GBM hồi quy rating (RMSE 0.849) tốt hơn MLP (0.868).
- **Item-CF** (rating của user trên 20 phim giống nhất theo co-rating) là đặc trưng mới có ích duy nhất: GBM "10 +
  item-CF" đạt AUC 0.797 / 0.837, +0.008 đến +0.011 trên cả nhãn tuyệt đối lẫn tương đối (vượt nhiễu bootstrap).
  content-kNN (embedding) và `peer_std` không giúp (xem [mục 9](#9-đặc-trưng-láng-giềng-cho-gbm-scriptseval_extra_features-đã-xoá)).
- **gbm_stack** (đang dùng trong `score_candidate`, [mục 11](#11-gbm_stack-đánh-giá-đầy-đủ-ngưỡng-theo-nhóm-tích-hợp-vào-score_candidate)) = GBM (10 đặc trưng + item-CF) + dự đoán MLP làm đặc trưng: AUC 0.801 /
  0.841, precision tổng ~81.5% ở độ phủ ~30% (gbm_13: 82.4% ở 23.0%; MLP: ~77% ở ~31%). MLP + item-CF có độ phủ cao
  nhất (~34%, precision ~78%) và RMSE thấp nhất trong các MLP (0.853–0.860) (xem [mục 10](#10-ứng-viên-gbm--item-cf-mlp--item-cf-stacking-mlp--gbm-scriptseval_candidates-đã-xoá)).
- Nút thắt của MLP không phải dung lượng mô hình. Hướng tiếp theo: attention trên lịch sử user (DIN), thêm tích
  phim ⊙ user (xem [Hướng tiếp theo](#13-hướng-tiếp-theo)).

## 1. Bài toán và dữ liệu

- `score_candidate(user_id, movie_id)` trả verdict **pass** (dự đoán user thích), **fail** (không thích) hoặc không
  có verdict (cần `llm_judge`).
- Nhãn: **thích** ⇔ rating ≥ 4.0, **không thích** ⇔ rating ≤ 2.5, còn lại là "ở giữa".
- Dữ liệu: `data/ml-latest-small-filtered/ranking_split/`, 610 user, 5135 phim.

| split | số rating | dùng cho |
|---|---|---|
| `ratings_train.csv` | 59,023 | mọi tín hiệu / đặc trưng / lịch sử user; train MLP |
| `ratings_validation.csv` | 7,377 | chọn ngưỡng và shrinkage; fit logreg/GBM; early stopping MLP |
| `ratings_test.csv` | 7,664 | chỉ làm nhãn khi đánh giá |

- Split **theo thời gian cho từng user**: mọi rating validation đều sau train, mọi rating test đều sau validation
  (đã kiểm tra: 100%). 4.2% cặp validation và 5.0% cặp test là phim chưa có rating nào trong train.
- Tỉ lệ nền trên toàn bộ test: thích 46.4% (3556), không thích 20.2% (1551), ở giữa 2557.
  Trên mẫu 2000 cặp: thích 45.5%, không thích 20.4%.

## 2. Config chung

`trusted_ai/taste/config.py` (`TasteConfig`), dùng cho luồng quy tắc và các đặc trưng của logreg/GBM:

| tham số | giá trị | ý nghĩa |
|---|---|---|
| `liked_rating` / `disliked_rating` | 4.0 / 2.5 | ngưỡng nhãn và ngưỡng verdict của luồng quy tắc |
| `similar_users_k` | 20 | top-k user tương tự trong số user đã rate phim |
| `min_similar_raters` | 3 | số rater tối thiểu để nhánh peer ra verdict |
| `min_common_ratings` | 5 | số phim chung tối thiểu khi tính độ tương tự user |
| `cf_shrinkage` | 2.0 | kéo dự đoán peer về user mean khi ít rater |
| `baseline_shrinkage` | 25.0 | shrinkage của movie bias |
| `avoid_deviation` | −0.5 | ngưỡng genre bị "né" |

Mục tiêu precision khi chọn ngưỡng cho mô hình học được (`scripts/experiment_score_model.py`): pass ≥ 80%, fail ≥ 70%.

Embedding phim: `Qwen/Qwen3-Embedding-4B`, 2560 chiều, văn bản `"{title} ({year}). Genres: {genres}. {plot}"`
(cắt ở 6000 ký tự, 350 phim bị cắt). Môi trường: torch 2.10.0, scikit-learn 1.8.0, chạy CPU.

### Lệnh tái lập

| lệnh | kết quả |
|---|---|
| `python -m scripts.eval_score_candidate` | `score_candidate` đang chạy (gbm_stack nếu có `data/score_model/`, không thì luồng B) trên 2000 cặp (seed 42) → `report.md`, `pairs.csv` |
| `python -m scripts.experiment_score_model [--n-test 0]` | B vs logreg vs GBM (2000 cặp, hoặc toàn bộ với `--n-test 0`) → `model_experiment.md` |
| `python -m scripts.eval_score_models_full` | kNN / logreg / GBM trên toàn bộ test → `model_eval_full.md`, `model_eval_full_pairs.csv` |
| `python -m scripts.eval_relative_labels [--seed 0]` | gbm_reg + MLP với nhãn / ngưỡng tương đối theo user_mean → `relative_labels.md` |
| `python -m scripts.train_score_model` | train gbm_stack cho `score_candidate` → `data/score_model/` |

Các file `.md` do script sinh ra là output thô; kết quả đã tổng hợp nằm trong note này. `scripts/mlp_score/` giờ chỉ
còn phần thư viện (dữ liệu, model, vòng train) mà `scripts.train_score_model` và `scripts.eval_relative_labels` dùng.

**Script đã xoá sau khi có kết luận (2026-09-28)** — số liệu của chúng chỉ còn trong note này (mục 7, 9, 10, 11.2):
`python -m scripts.mlp_score` (CLI thí nghiệm MLP, kể cả `--item-cf`), `scripts.eval_extra_features`,
`scripts.eval_candidates`, `scripts.eval_group_thresholds`. Code item-CF cho MLP (leave-one-out trên hàng
similarity) cũng đã xoá vì MLP + item-CF không được dùng; cách làm được mô tả ở mục 10.

## 3. Các model đã đánh giá

| tên | loại | fit trên | đầu vào | đầu ra → verdict |
|---|---|---|---|---|
| **Luồng cũ** | quy tắc | – | trung bình rating thô của peer; content percentile (pass ≥ p80, fail ≤ p20); genre né | pass / fail / neutral theo tầng |
| **B** (fallback) | quy tắc theo tầng | – | peer kNN mean-centred → baseline (user mean + movie bias) → genre né | tầng đầu có dự đoán ≥ 4.0 / ≤ 2.5 quyết định; content chỉ ghi evidence |
| **kNN** | tầng peer của B đứng riêng | – | top-20 user tương tự đã rate phim, ≥ 3 rater | dự đoán ≥ 4.0 pass, ≤ 2.5 fail; phủ 84.2% số cặp |
| **bias** | baseline | train | mu + b_u + b_i (shrinkage user 5, phim 25) | chỉ dùng để so AUC / RMSE |
| **logreg** | hồi quy logistic | 7377 cặp validation | 13 đặc trưng (bên dưới), chuẩn hoá | P(thích), P(không thích); ngưỡng chọn out-of-fold (GroupKFold theo user) |
| **GBM** | `HistGradientBoostingClassifier` (200 vòng, lr 0.05, 15 lá, ≥ 40 mẫu/lá) | 7377 cặp validation | 13 đặc trưng | như logreg |
| **MLP** | mạng nơ-ron, `scripts/mlp_score/` | 59,023 rating train, che phim mục tiêu | vector phim + vector user thưa | dự đoán rating; ngưỡng chọn trên validation |

**13 đặc trưng của logreg/GBM** (đều tính từ train):
- peer: `peer_offset`, `log_n_raters`, `has_peer`
- baseline: `user_mean`, `movie_bias`, `log_movie_n`
- user: `user_std`, `log_user_n`
- genre: `genre_dev_mean`, `genre_dev_min`, `avoided_hit`
- content: `content_percentile`, `content_pos_minus_neg`

**Config MLP mặc định:**

| thành phần | giá trị |
|---|---|
| vector phim | PCA 128 chiều (whiten) của embedding 2560 chiều, giữ 54.9% phương sai; tuỳ chọn `--genres` nối thêm one-hot 19 genre |
| vector user | lưu thưa (vị trí phim, giá trị), giá trị = (rating − mu − b_u) / sqrt(n) |
| nhánh user | `EmbeddingBag(5135, 64)` học được (tương đương `Linear(5135 → 64)`) + hồ sơ nội dung = cùng tổng có trọng số trên vector phim cố định |
| MLP | nối [phim, user_cf, user_content] → 256 → 64 → 1, ReLU, dropout 0.3 |
| mục tiêu | phần dư rating − (mu + b_u + b_i), loss MSE; dự đoán được cắt về [0.5, 5] |
| chống rò rỉ | khi train: che phim mục tiêu khỏi lịch sử; b_u và giá trị center tính **không có** rating mục tiêu (nếu dùng mean đầy đủ, tổng lịch sử còn lại bằng đúng −(r − mean)) |
| denoising | bỏ ngẫu nhiên thêm 20% rating khác của user khi train |
| tối ưu | AdamW lr 1e-3, weight decay 1e-4, batch 256, tối đa 60 epoch, early stopping patience 6 theo val RMSE |
| tham số học được | 427,329 |

## 4. Luồng quy tắc: cũ vs B (mẫu 2000 cặp test, seed 42, 480 user)

| | luồng cũ | B (hiện tại) |
|---|---|---|
| có verdict | 1380 (69.0%) | 709 (35.4%) |
| không verdict (cần `llm_judge`) | 620 (31.0%) | 1291 (64.5%) |
| accuracy chỉ tính pass/fail | 51.6% trên 876 cặp | **72.8%** trên 709 cặp |

Precision theo nhánh của **B**:

| nhánh | verdict | n | precision | tỉ lệ nền | lift |
|---|---|---|---|---|---|
| `similar_user_pseudo_label` | pass | 443 | 79.5% | 45.5% | 1.75x |
| `similar_user_pseudo_label` | fail | 117 | 74.4% | 20.4% | 3.64x |
| `baseline_prediction` | pass | 51 | 80.4% | 45.5% | 1.77x |
| `baseline_prediction` | fail | 13 | 69.2% | 20.4% | 3.39x |
| `genre_deviation` | fail | 85 | **31.8%** | 20.4% | 1.55x |

Ở luồng cũ, content ra verdict cho 546 cặp nhưng pass chỉ đúng 44.8% (ngang tỉ lệ nền 45.5%), còn neutral của peer đúng
34.5%. Đó là lý do content bị bỏ khỏi verdict: nội dung giống nhau dự đoán *user sẽ xem* hơn là *user sẽ thích*.

**Độ nhạy của từng tín hiệu** (trên mẫu 2000 cặp):

| tín hiệu | n | Pearson r | AUC thích |
|---|---|---|---|
| trung bình thô của peer (≥ 1 rater) | 1881 | 0.403 | 0.705 |
| `peer_predicted_rating` (≥ 1 rater) | 1881 | 0.561 | 0.783 |
| `peer_predicted_rating` (≥ 3 rater) | 1689 | 0.556 | 0.776 |
| `baseline_predicted_rating` (chỉ phần peer không quyết định) | 1440 | 0.310 | 0.652 |
| `content_percentile` | 1999 | 0.156 | 0.598 |

Tỉ lệ thích tăng đều theo content percentile (29.7% ở p0–20 → 52.7% ở p80–100): có tín hiệu, nhưng quá yếu để tự ra
verdict.

## 5. Mô hình học được vs B

### 5.1 AUC

| model | đặc trưng | mẫu 2000: thích | mẫu 2000: không thích | toàn bộ test: thích | toàn bộ test: không thích |
|---|---|---|---|---|---|
| B: `peer_predicted` | 1 | 0.781 | 0.783 | 0.762 | 0.788 |
| logreg | peer + baseline (6) | 0.784 | 0.785 | 0.763 | 0.788 |
| logreg | bỏ genre (10) | 0.792 | 0.805 | 0.772 | 0.812 |
| logreg | bỏ content (11) | 0.795 | 0.807 | 0.776 | 0.810 |
| logreg | cả 13 | 0.798 | 0.812 | 0.779 | 0.817 |
| GBM | cả 13 | **0.806** | **0.820** | **0.789** | **0.828** |

Theo độ lớn hệ số logreg, đặc trưng mạnh nhất là `user_mean` (+0.84 / −0.71), `peer_offset` (+0.66 / −0.64) và
`content_pos_minus_neg` (+0.20 / −0.47, đặc biệt quan trọng cho chiều không thích).

### 5.2 Cùng độ phủ với B (toàn bộ test, lấy top-k theo xác suất với k = số verdict của B)

| | k | B | logreg | GBM |
|---|---|---|---|---|
| pass | 1864 | 76.9% ±1.9 | 78.2% ±1.9 | 78.5% ±1.9 |
| fail (không tính genre) | 506 | 66.8% ±4.1 | 70.6% ±4.0 | 80.0% ±3.5 |
| fail (gồm genre) | 852 | 53.8% ±3.3 | 66.8% ±3.2 | 70.7% ±3.1 |

Trên toàn bộ test, B ra 2716 verdict (35.4%) với precision tổng 69.6% ±1.7. Chỗ B kém nhất là fail, do nhánh genre kéo
xuống.

## 6. Toàn bộ test: kNN vs logreg vs GBM vs MLP (7664 cặp, 610 user)

MLP ở bảng này là cấu hình mặc định, seed 0 (epoch tốt nhất 11 / 17).

### 6.1 Verdict theo ngưỡng

| model | ngưỡng pass | pass n | precision pass | recall thích | ngưỡng fail | fail n | precision fail | recall không thích | độ phủ | precision tổng |
|---|---|---|---|---|---|---|---|---|---|---|
| kNN | pred ≥ 4.0 | 1652 | 77.8% | 36.2% | pred ≤ 2.5 | 435 | 67.8% | 19.0% | 27.2% | 75.8% |
| logreg | P ≥ 0.691 | 1648 | 79.3% | 36.8% | P ≥ 0.622 | 456 | 71.3% | 21.0% | 27.5% | 77.6% |
| GBM | P ≥ 0.740 | 1453 | **81.5%** | 33.3% | P ≥ 0.729 | 307 | **86.6%** | 17.2% | 23.0% | **82.4%** |
| MLP | pred ≥ 3.931 | 1824 | 79.1% | **40.6%** | pred ≤ 2.493 | 568 | 73.8% | **27.0%** | **31.2%** | 77.8% |

Ở cùng recall với MLP, precision của các model khác là:

| | precision khi recall thích = 40.6% | precision khi recall không thích = 27.0% |
|---|---|---|
| logreg | 78.2% | 70.1% |
| GBM | 78.6% | **79.1%** |
| MLP | **79.1%** | 73.8% |

→ Độ phủ cao hơn của MLP chủ yếu đến từ ngưỡng lỏng hơn. Ở chiều thích MLP ngang GBM, ở chiều không thích MLP kém GBM.

### 6.2 AUC và sai số

| model | AUC thích | AUC không thích | AUC thích (tập kNN, 6456 cặp) | AUC không thích (tập kNN) | RMSE test |
|---|---|---|---|---|---|
| kNN | – | – | 0.758 | 0.784 | – |
| bias | 0.738 | 0.758 | 0.733 | 0.752 | 0.932 |
| logreg | 0.779 | 0.817 | 0.774 | 0.813 | – |
| GBM | **0.789** | **0.828** | **0.782** | **0.819** | – |
| MLP | 0.774 | 0.807 | 0.771 | 0.801 | **0.868** |

### 6.3 Precision tại mức recall cố định

| model | thích @10% | @25% | @50% | @75% | không thích @10% | @25% | @50% | @75% |
|---|---|---|---|---|---|---|---|---|
| kNN | 87.9% | 82.7% | 72.2% | 61.2% | 76.1% | 59.4% | 37.9% | 20.1% |
| bias | 87.9% | 80.6% | 71.5% | 61.1% | 68.4% | 65.4% | 48.5% | 32.5% |
| logreg | 89.7% | 83.9% | 74.5% | 65.9% | 80.8% | 70.8% | 60.4% | 41.6% |
| GBM | 90.1% | **84.3%** | **75.8%** | **66.7%** | **93.4%** | **80.8%** | **63.1%** | **42.8%** |
| MLP | **92.0%** | 84.1% | 75.3% | 64.3% | 83.9% | 75.8% | 59.2% | 40.0% |

## 7. Các biến thể MLP

### 7.1 Số chiều và one-hot genre

Trung bình 2 seed (0, 1) trên toàn bộ test. "prec. ¬thích @25%" = precision chiều không thích ở recall 25%.

| PCA / user | genre | tham số | epoch tốt nhất | val RMSE | test RMSE | AUC thích | AUC không thích | prec. ¬thích @25% | độ phủ | precision tổng |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 / 256 | – | 1.66M | 7 | 0.8502 | 0.8802 | 0.772 | 0.802 | 71.9% | 28.4% | 78.2% |
| 512 / 64 | – | 0.62M | 7.5 | 0.8517 | 0.8824 | 0.770 | 0.802 | 69.2% | 28.8% | 77.4% |
| 128 / 256 | – | 1.46M | 8.5 | 0.8461 | 0.8771 | 0.770 | 0.807 | 73.9% | 31.8% | 76.9% |
| 128 / 64 | – | 0.43M | 12 | 0.8451 | 0.8698 | 0.775 | 0.809 | **75.4%** | 31.8% | 77.0% |
| **128 / 64** | **có** | 0.44M | 10 | **0.8395** | **0.8658** | **0.778** | **0.814** | 73.3% | **32.8%** | 76.7% |
| 64 / 64 | – | 0.39M | 12.5 | 0.8443 | 0.8770 | 0.774 | 0.802 | 68.1% | 31.2% | 75.9% |
| 64 / 32 | – | 0.22M | 16.5 | 0.8458 | 0.8756 | 0.775 | 0.803 | 71.3% | 32.4% | 75.9% |
| 64 / 32 | có | 0.23M | 14 | 0.8440 | 0.8715 | 0.777 | 0.809 | 69.7% | 31.9% | 76.0% |
| 32 / 16 | – | 0.12M | 21 | 0.8441 | 0.8737 | 0.773 | 0.803 | 70.2% | 29.9% | 77.4% |
| 32 / 16 | có | 0.13M | 15.5 | 0.8440 | 0.8731 | 0.774 | 0.804 | 69.9% | 30.9% | 76.6% |

- **To hơn thì overfit:** epoch tốt nhất đến sớm (6–9), RMSE tệ hơn. Phần phương sai mà PCA 512 giữ thêm chủ yếu là
  chi tiết cốt truyện, không giúp dự đoán gu.
- **Nhỏ hơn thì thiếu thông tin**, rõ nhất ở chiều không thích. 128 / 64 là điểm cân bằng.
- **Genre** giúp đều nhưng rất ít (+0.001 đến +0.006 AUC), cỡ độ dao động giữa các seed.

### 7.2 Embedding có chứa genre không?

Genre có trong văn bản được embed, nhưng chỉ nằm ngầm. Dùng logistic regression dự đoán từng genre từ embedding (AUC,
5-fold):

| genre | số phim | embedding đầy đủ | PCA 128 |
|---|---|---|---|
| Drama | 2326 | 0.918 | 0.894 |
| Comedy | 2117 | 0.974 | 0.965 |
| Thriller | 1047 | 0.938 | 0.926 |
| Action | 953 | 0.958 | 0.943 |
| Romance | 990 | 0.931 | 0.917 |
| Horror | 492 | 0.989 | 0.981 |
| Documentary | 82 | 0.989 | 0.993 |
| Animation | 231 | 0.979 | 0.955 |

Genre rõ nét gần như còn nguyên sau PCA; genre mơ hồ (Drama) mất nhiều hơn. Điều này giải thích vì sao thêm one-hot
genre chỉ giúp ít.

## 8. Nhãn tương đối theo trung bình user (`scripts.eval_relative_labels`)

Câu hỏi: ngưỡng tuyệt đối (≥ 4.0 / ≤ 2.5) có bất lợi cho user khó tính không? Nhãn thử: **thích** ⇔ rating −
user_mean ≥ +0.5, **không thích** ⇔ ≤ −0.5 (user_mean = trung bình train). Hai model dự đoán rating: **gbm_reg**
(`HistGradientBoostingRegressor`, 13 đặc trưng, fit trên validation) và MLP mặc định (seed 0). Toàn bộ test, 7664 cặp.

### 8.1 Nhãn tương đối khớp với rating thật đến đâu

| nhóm (ngũ phân vị user_mean) | thích tuyệt đối | thích tương đối | thích tương đối có rating ≥ 4 | không thích tuyệt đối | không thích tương đối | không thích tương đối có rating ≤ 2.5 (rating TB) |
|---|---|---|---|---|---|---|
| Q1 khó tính (1.4–3.2) | 25.3% | 29.9% | 84.6% | 41.3% | 34.8% | 100% (1.58) |
| Q2 (3.2–3.42) | 37.7% | 37.7% | 100% | 19.6% | 19.6% | 100% (1.94) |
| Q3 | 44.2% | 25.0% | 100% | 20.0% | 35.9% | 55.7% (2.32) |
| Q4 | 54.1% | 22.6% | 100% | 11.2% | 34.1% | 33.0% (2.65) |
| Q5 dễ tính (3.91–5.0) | 71.4% | 26.6% | 100% | 8.0% | 26.6% | 29.9% (2.78) |
| tổng | 46.4% | 28.3% | **96.6%** | 20.2% | 30.3% | **62.2%** |

- "Thích" tương đối gần như luôn là rating ≥ 4. Riêng Q1 có 15% là 3.0–3.5. Tỉ lệ thích cân bằng giữa các nhóm
  (22–38% thay vì 25–71%).
- "Không thích" tương đối ở user dễ tính phần lớn là rating 3.0–3.5 (rating TB 2.65–2.78).
- 1443 cặp thích tuyệt đối thành "ở giữa" (rating 4.0 của user dễ tính); 861 cặp ở giữa thành "không thích".

### 8.2 Dự đoán

| model | RMSE test | AUC thích (tương đối) | AUC không thích (tương đối) | AUC thích (tuyệt đối) | AUC không thích (tuyệt đối) |
|---|---|---|---|---|---|
| gbm_reg | **0.8485** | **0.737** | **0.759** | **0.786** | **0.824** |
| MLP | 0.8684 | 0.711 | 0.734 | 0.774 | 0.807 |
| bias | 0.9321 | 0.672 | 0.689 | 0.738 | 0.758 |

- Nhãn tương đối khó hơn nhiều (AUC giảm 0.05–0.07): phần lớn sức mạnh của các model là biết user nào dễ tính
  (`user_mean` là đặc trưng quan trọng nhất theo permutation importance), nhãn tương đối bỏ tín hiệu đó đi.
- gbm_reg hơn MLP cả RMSE lẫn AUC, dù chỉ fit trên 7377 cặp validation.

### 8.3 Verdict

| model | luật | pass n | pass đúng (tương đối) | pass có rating ≥ 4 | fail n | fail đúng (tương đối) | fail có rating ≤ 2.5 | độ phủ | user có ≥ 1 pass |
|---|---|---|---|---|---|---|---|---|---|
| gbm_reg | tuyệt đối 4.0 / 2.5 | 1436 | 40.5% | 81.7% | 696 | 67.4% | 74.7% | 27.8% | 368 / 610 |
| gbm_reg | lệch ±0.5 | 671 | 67.4% | 69.6% | 1201 | 66.2% | 55.8% | 24.4% | 208 / 610 |
| gbm_reg | lệch ±0.75 | 227 | 78.9% | 75.3% | 616 | 73.5% | 64.4% | 11.0% | 89 / 610 |
| gbm_reg | lệch ±1.0 | 82 | 89.0% | 79.3% | 261 | 76.2% | 69.0% | 4.5% | 33 / 610 |
| MLP | tuyệt đối 4.0 / 2.5 | 1544 | 40.1% | 80.6% | 576 | 66.1% | 73.6% | 27.7% | 347 / 610 |
| MLP | lệch ±0.5 | 711 | 63.0% | 73.7% | 957 | 66.1% | 51.2% | 21.8% | 198 / 610 |

Chọn ngưỡng trên validation cho precision pass (tương đối) ≥ 80% không đạt ở ngưỡng nào → không ra pass.

Theo nhóm (gbm_reg):

| nhóm | luật | pass n | pass đúng (tương đối) | pass có rating ≥ 4 | độ phủ | user có ≥ 1 pass |
|---|---|---|---|---|---|---|
| Q1 khó tính | tuyệt đối | 43 | 93.0% | 83.7% | 36.5% | 19 / 88 |
| Q1 khó tính | lệch ±0.5 | 309 | 70.9% | 61.2% | 43.9% | 55 / 88 |
| Q5 dễ tính | tuyệt đối | 917 | 32.5% | 82.8% | 59.9% | 178 / 191 |
| Q5 dễ tính | lệch ±0.5 | 19 | 63.2% | 84.2% | 9.5% | 15 / 191 |

- Ngưỡng tuyệt đối đúng là gần như không ra pass cho user khó tính (19 / 88 user), còn pass của user dễ tính phần lớn
  là rating 4.0 "bình thường" với họ.
- Ngưỡng tương đối chuyển pass từ user dễ tính sang user khó tính, nhưng precision chỉ khoảng 63–71% và tổng số user
  có pass giảm (208 so với 368), vì model hiếm khi dự đoán user dễ tính cho ≥ 4.5.

## 9. Đặc trưng láng giềng cho GBM (`scripts.eval_extra_features`, đã xoá)

Thêm vào GBM phân loại (fit trên validation) các đặc trưng "user đã chấm những phim giống phim này ra sao", k = 20,
tính từ train:
- **content-kNN** (cosine embedding Qwen): `content_knn_offset` = trung bình có trọng số của rating − user_mean trên
  20 phim giống nhất user đã chấm; `max_sim_liked` / `max_sim_disliked` = độ giống cao nhất với phim user chấm
  ≥ mean + 0.5 / ≤ mean − 0.5.
- **item-CF** (adjusted cosine trên rating train, shrinkage n / (n + 10)): `item_knn_offset` (như trên),
  `item_knn_weight` (tổng độ giống của top-20).
- **`peer_std`**: std của (rating − mean của peer) trong các user tương tự đã chấm phim (≥ 2 người).
- "10" = 13 đặc trưng trừ `has_peer`, `avoided_hit`, `content_percentile` (importance ≈ 0).

"AUC trong user" = trung bình AUC tính riêng từng user: đo khả năng xếp hạng phim cho cùng một người.

| bộ đặc trưng | AUC thích / không thích (tuyệt đối) | AUC thích / không thích (tương đối) | AUC trong user, thích (tương đối) | prec@25% thích / không thích (tuyệt đối) |
|---|---|---|---|---|
| 13 hiện tại | 0.789 / 0.828 | 0.762 / 0.782 | 0.660 | 84.3% / 80.8% |
| 10 (bỏ 3) | 0.791 / 0.830 | 0.764 / 0.784 | 0.653 | 84.7% / 80.2% |
| 10 + peer_std | 0.788 / 0.830 | 0.765 / 0.782 | 0.669 | 85.1% / 80.7% |
| 10 + content-kNN | 0.789 / 0.828 | 0.764 / 0.783 | 0.664 | 84.7% / 78.4% |
| **10 + item-CF** | **0.797 / 0.837** | **0.773 / 0.792** | 0.673 | **86.1% / 82.6%** |
| 10 + tất cả | 0.794 / 0.833 | 0.772 / 0.788 | 0.677 | 85.1% / 79.7% |

Bootstrap 200 lần theo user, "10 + item-CF" so với "13 hiện tại": Δ AUC tổng +0.008 đến +0.011 trên cả 4 mục tiêu, khoảng
95% đều > 0 (ví dụ thích tuyệt đối +0.0081 [+0.0048, +0.0114]). Δ AUC trong user +0.004 đến +0.014 nhưng khoảng
95% cắt 0 (±0.02: nhiều user chỉ có vài cặp test).

- **item-CF là tín hiệu mới duy nhất có ích.** Trong bộ "10 + tất cả", `item_knn_offset` là đặc trưng quan trọng thứ
  hai theo AUC tổng (sau `user_mean`) và **quan trọng nhất theo AUC trong user** (giảm 0.04–0.08 khi xáo).
- **content-kNN và `peer_std` không giúp** (chênh lệch ≤ 0.002, cỡ nhiễu). Khớp với nhận xét cũ: độ giống nội dung
  văn bản dự đoán "sẽ xem" hơn là "sẽ thích".
- Bỏ 3 đặc trưng không mất gì. Thêm tất cả lại kém hơn chỉ thêm item-CF: 7377 cặp validation không đủ cho thêm đặc
  trưng nhiễu.
- AUC trong user chỉ khoảng 0.65–0.68: xếp hạng phim cho cùng một người vẫn là phần khó nhất.

## 10. Ứng viên: GBM + item-CF, MLP + item-CF, stacking MLP → GBM (`scripts.eval_candidates`, đã xoá)

Nhãn tuyệt đối, toàn bộ test (7664 cặp), MLP chạy 2 seed (0, 1).
- **gbm_itemcf**: 10 đặc trưng (bỏ `has_peer`, `avoided_hit`, `content_percentile`) + `item_knn_offset`,
  `item_knn_weight`; fit trên validation, ngưỡng out-of-fold như GBM cũ.
- **mlp_itemcf**: MLP mặc định + 3 input (item_knn_offset, cờ thiếu, tổng độ giống). Với cặp train, rating mục tiêu
  được bỏ ra khỏi mọi chỗ: lịch sử, user mean và **hàng similarity của phim mục tiêu** (tích vô hướng, chuẩn, số người
  cùng chấm). Bản đầu chỉ che lịch sử: đặc trưng tương quan 0.71 với phần dư trên train so với 0.31 trên validation
  (ít người cùng chấm thì chính rating mục tiêu quyết định phim nào "giống"), MLP học tín hiệu rò rỉ và kém hơn MLP
  gốc (RMSE 0.893). Sau khi sửa: train 0.33, validation 0.31, test 0.34.
- **gbm_stack**: gbm_itemcf + `mlp_pred` từ MLP mặc định train đúng 11 epoch **không nhìn validation**, nên dự đoán
  trên validation là out-of-sample với GBM.

| model | RMSE test | AUC thích / không thích | AUC trong user thích / không thích (TB 2 seed) | pass n (prec.) | fail n (prec.) | độ phủ | precision tổng |
|---|---|---|---|---|---|---|---|
| gbm_13 (cũ) | – | 0.789 / 0.828 | 0.662 / 0.667 | 1453 (81.5%) | 307 (86.6%) | 23.0% | **82.4%** |
| gbm_itemcf | – | 0.797 / 0.837 | 0.676 / 0.679 | 1725 (81.2%) | 435 (81.8%) | 28.2% | 81.3% |
| mlp (seed 0 / 1) | 0.868 / 0.871 | 0.774–0.776 / 0.807–0.811 | 0.656 / 0.667 | 1824–1877 (77.5–79.1%) | 568–601 (72.0–73.8%) | 31.2–32.3% | 76.2–77.8% |
| mlp_itemcf (seed 0 / 1) | **0.860 / 0.853** | 0.784–0.789 / 0.815–0.821 | 0.669 / 0.677 | 1959–1980 (79.2–79.3%) | 593–659 (73.3–73.4%) | **33.6–34.2%** | 77.8–77.9% |
| **gbm_stack** (seed 0 / 1) | – | **0.801–0.802 / 0.841** | **0.678 / 0.681** | 1817–1823 (80.5–80.7%) | 459–495 (83.6–85.8%) | 29.8–30.2% | 81.2–81.7% |

Precision tại recall cố định (thích @25% / không thích @25%): gbm_13 84.3% / 80.8%, gbm_itemcf 86.1% / 82.6%,
mlp 83.2–84.1% / 75.0–75.8%, mlp_itemcf 85.2–86.8% / 74.5–74.6%, gbm_stack **86.2–86.8% / 84.0–86.6%**.

- **gbm_stack tốt nhất ở gần mọi chỉ số xếp hạng**, ổn định giữa 2 seed. So với gbm_13: độ phủ 23.0% → ~30% (thêm khoảng
  540 verdict) mà precision tổng chỉ giảm từ 82.4% xuống ~81.5%. So với MLP: gần cùng độ phủ, precision tổng cao hơn
  khoảng 4 điểm, chủ yếu ở fail (84–86% so với 72–74%). Xáo `mlp_pred` làm AUC giảm 0.09–0.12: GBM dựa nhiều vào nó.
- **item-CF giúp cả hai loại model**: gbm +0.008 / +0.009 AUC; MLP +0.010 đến +0.013 AUC và RMSE giảm 0.009–0.019.
- **mlp_itemcf có độ phủ cao nhất (~34%)** với precision tổng ~78% (ngang MLP cũ), nhưng chiều không thích vẫn yếu hơn
  GBM rõ rệt.

## 11. gbm_stack: đánh giá đầy đủ, ngưỡng theo nhóm, tích hợp vào `score_candidate`

### 11.1 Đánh giá đầy đủ (nhãn tuyệt đối, toàn bộ test)

- Hiệu chỉnh xác suất tốt: mọi khoảng 0.1 lệch ≤ 0.04 so với tỉ lệ thật (P(thích) ≈ 0.85 → 84% thích thật).
- Lỗi nặng hiếm: pass sai chủ yếu là rating 3–3.5 (286 / 1823); chỉ 66 pass (3.6%) là phim ≤ 2.5. Fail trúng phim
  ≥ 4 chỉ 12 / 459.
- Precision tại recall 10 / 25 / 50 / 75%: thích 90–92 / 86–87 / 77 / 68%; không thích 93–95 / 84–87 / 65–66 / 45–46%.

Theo nhóm user (ngũ phân vị user_mean, seed 0):

| nhóm | user_mean | AUC thích / không thích | pass n (prec.) | recall thích | fail n (prec.) | độ phủ | precision tổng | user có ≥ 1 pass |
|---|---|---|---|---|---|---|---|---|
| Q1 khó tính | 1.40–3.20 | 0.836 / 0.863 | 51 (86.3%) | 10.9% | 389 (87.9%) | 27.5% | 87.7% | **17 / 88** |
| Q2 | 3.20–3.42 | 0.729 / 0.796 | 100 (78.0%) | 14.1% | 34 (79.4%) | **9.1%** | 78.4% | 37 / 86 |
| Q3 | 3.42–3.63 | 0.768 / 0.807 | 270 (78.1%) | 30.8% | 34 (67.6%) | 19.7% | 77.0% | 64 / 111 |
| Q4 | 3.63–3.91 | 0.729 / 0.760 | 391 (77.7%) | 37.0% | 2 (100%) | 25.9% | 77.9% | 97 / 134 |
| Q5 dễ tính | 3.91–5.00 | 0.748 / 0.766 | 1011 (82.5%) | 76.3% | 0 | 66.0% | 82.5% | 186 / 191 |

- User khó tính: pass rất đáng tin (86% tuyệt đối, 94% theo gu riêng, rating TB 4.25) nhưng hiếm: 71 / 88 user không
  có pass nào. Phần lớn verdict của nhóm là fail (đúng 88% tuyệt đối, nhưng chỉ 71% theo gu riêng).
- 55% số pass dồn vào user dễ tính; nhóm Q2 gần như không có verdict.

### 11.2 Ngưỡng theo nhóm (`scripts.eval_group_thresholds`, đã xoá)

Chọn ngưỡng riêng cho từng ngũ phân vị user_mean (mốc từ validation), trên xác suất out-of-fold, cùng mục tiêu
precision pass ≥ 80% / fail ≥ 70%.

| | pass n (prec.) | fail n (prec.) | độ phủ | precision tổng | user có ≥ 1 pass | pass cho Q1 khó tính |
|---|---|---|---|---|---|---|
| ngưỡng chung | 1823 (80.7%) | 459 (85.8%) | 29.8% | **81.7%** | **401 / 610** | 51 (86.3%) |
| ngưỡng theo nhóm | 1755 (79.3%) | 620 (76.8%) | 31.0% | 78.6% | 334 / 610 | **0** |

- **Không giúp, còn làm tệ hơn.** Q1 và Q2 không đạt precision pass 80% trên validation (tỉ lệ thích nền 20% và 35%),
  nên ngưỡng pass thành vô cực và user khó tính mất cả 51 pass. Q5 được ngưỡng thấp hơn (0.569) nên pass càng dồn về
  user dễ tính. → Giữ ngưỡng chung. Muốn thêm pass cho user khó tính phải chấp nhận mục tiêu precision thấp hơn
  cho nhóm đó (quyết định sản phẩm), hoặc xếp hạng top-k theo từng user.

### 11.3 Tích hợp (`trusted_ai/taste/learned/`, `scripts.train_score_model`)

- `CandidateScorer.score`: có `data/score_model/` và user có trong train → verdict của gbm_stack
  (`source = learned_model`); không đủ tự tin → `llm_judge`. Không có model / user lạ → luồng rule cũ.
- MLP được tính sẵn thành ma trận 610 × 5135 (`mlp_pred.npz`), nên lúc chạy không cần torch. Ma trận được kiểm tra
  khớp `predict()` (sai lệch ≤ 1e-4) ngay trong script train.
- Kiểm tra: đặc trưng sau khi chuyển code giống hệt trước (max |Δ| = 0 trên 1500 cặp); `TasteService.score_candidate`
  trên 7664 cặp test cho **1823 pass (80.7%), 459 fail (85.8%), độ phủ 29.8%**, trùng thí nghiệm. Khởi động
  `TasteService` ~3.5 s, mỗi lần `score()` ~14 ms. 32 unit test qua (4 test mới cho nhánh model).

## 12. Hạn chế

- Mỗi cấu hình MLP chỉ chạy 1–2 seed. Chênh lệch AUC dưới khoảng 0.005 nên coi là nhiễu.
- MLP: split theo thời gian, nhưng khi train phim bị che là phim ngẫu nhiên trong quá khứ (lệch phân phối). Validation
  vừa dùng cho early stopping vừa để chọn ngưỡng, nên ngưỡng có thể hơi lạc quan.
- MLP: b_i trong baseline leave-one-out vẫn thấy rating mục tiêu qua b_u (ảnh hưởng bậc hai, trọng số 1 / (n_u + 5)).
- logreg/GBM chỉ fit trên 7377 cặp validation, vì đặc trưng phải tính từ train.
- Bảng 5.2 lấy số của B từ báo cáo làm tròn. Số verdict đúng của B (thay vì precision) không được lưu cho toàn bộ test.
- Mẫu 2000 cặp và toàn bộ test cho AUC lệch nhau khoảng 0.02, nên chỉ so các model trong cùng một tập.
- gbm_stack: MLP train đúng 11 epoch không nhìn validation, nhưng con số 11 lấy từ lần chạy early stopping trên
  validation trước đó (ảnh hưởng nhỏ). GBM chỉ chạy một random seed.
- Item-CF cho MLP: đã bỏ rating mục tiêu khỏi hàng similarity của phim mục tiêu, nhưng user mean dùng để centre các
  rating khác của user trong ma trận vẫn gồm rating mục tiêu (bậc hai, trọng số 1 / n_u).

## 13. Hướng tiếp theo

Chưa làm; cần duyệt trước.

1. **Attention trên lịch sử user (DIN):** vector user phụ thuộc phim mục tiêu, nghiêng về các phim giống nó trong lịch
   sử. Có thể giúp chiều không thích (tín hiệu "ghét 3 phim kinh dị" không còn bị pha loãng trong 100 phim), và các
   trọng số attention giải thích được dự đoán.
2. **Tích phim ⊙ hồ sơ nội dung user** làm đầu vào: MLP học phép nhân rất kém. Nên làm cùng hướng 1 để tách được hiệu
   quả của phép nhân với hiệu quả của attention.
3. ~~Dự đoán MLP làm đặc trưng cho GBM~~: đã làm ở mục 10 (gbm_stack). Có thể thử stack bằng mlp_itemcf thay vì MLP gốc.

## 14. File trong folder

| file | nội dung |
|---|---|
| `experiment_note.md` | note này |
| `regression_models.md` | output thô của `scripts.eval_regression_models` (không thuộc các thí nghiệm trong note này) |

Các script ghi output thô vào folder này khi chạy (xem [lệnh tái lập](#lệnh-tái-lập)).
