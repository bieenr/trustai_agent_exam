# Kết quả: agent trên `recommend_test.json` (2026-09-28, cập nhật 2026-09-29)

Đo **ràng buộc cứng** và **độ hợp ngữ cảnh (semantic, LLM judge; mục 1–4: `gpt-4.1-mini`, mục 5–6: `deepseek-v4.1-flash`)**; taste của `score_candidate` đã
đánh giá riêng trên toàn bộ test split (`evaluation/score_candidate/experiment_note.md`).

**File trong thư mục** (cập nhật 2026-09-29): chỉ giữ lượt chạy hiện tại — agent hiện tại + judge `deepseek-v4.1-flash`,
luồng `pass_first` (mục 6):
- `agent_pass_first/`: transcript từng case (câu trả lời, lời gọi tool, danh sách phim, độ trễ) — đầu vào của
  `summarize_agent_eval`.
- `agent_pass_first_recs.json`: `{case_id: [movie_id, ...]}`, danh sách phim hiển thị cho user — đầu vào của bộ chấm.
- `eval_pass_first.json`: kết quả chấm từng phim theo ràng buộc cứng / semantic / taste và metric từng case.
- `summary.md`: bảng tổng hợp + danh sách phim trượt, sinh từ hai nguồn trên.
- `agent_pass_first.json`: định dạng cũ (một file) của lượt chạy ở mục 5, giữ lại để tham khảo.

Các lượt so sánh ở mục 1–4 và dòng "trước sửa tool" ở mục 5 đã xoá file; số liệu chỉ còn trong file này và `REPORT.md`.

## Tái lập

```bash
PY=python  # môi trường có langchain (requirements.txt) + scikit-learn
$PY -m scripts.run_agent_cases --verdict-filter off|pass_first|pass_only   # agent → agent_<f>/ (transcript), agent_<f>_recs.json (cache theo case)
$PY -m scripts.run_conversation_eval evaluation/recommendation/results/agent_<f>_recs.json -k 10 \
    --output evaluation/recommendation/results/eval_<f>.json               # judge cache ở .eval_cache/
python -m scripts.summarize_agent_eval off pass_first pass_only            # → summary.md
```

`verdict_filter` (TasteConfig): tool lấy 20 ứng viên, chấm bằng `score_candidate`, rồi `off` = giữ thứ hạng cũ,
`pass_first` (mặc định) = bỏ fail, phim pass trước, bù bằng phim chưa chắc; `pass_only` = chỉ phim pass.

## 1. Agent chạy thật (60 case, mỗi luồng một lượt chạy agent)

| luồng | case có phim | case 0 phim | ràng buộc: phim đạt | semantic: phim đạt | semantic: hit rate |
|---|---|---|---|---|---|
| off | 53 / 60 | 7 | 67.6% | 54.3% | 69.0% |
| pass_first | 56 / 60 | 4 | 66.1% | 59.9% | 71.0% |
| pass_only | 43 / 60 | 17 | 61.8% | 51.6% | 62.5% |

Chênh lệch giữa các lượt chạy agent **lẫn cả nhiễu của LLM** (mỗi lượt agent chọn tool/tham số khác nhau).

## 2. Phát lại đúng lời gọi tool của lượt `off` (tách riêng tác động của bộ lọc)

| luồng | phim / case | case 0 phim | ràng buộc: phim đạt | semantic: phim đạt | semantic: hit rate |
|---|---|---|---|---|---|
| off | 3.83 | 7 | 63.5% | 53.3% | 72.4% |
| pass_first | 3.83 | 7 | 64.8% | 54.0% | 72.4% |
| pass_only | 2.38 | 20 | 62.9% | 48.5% | 69.6% |

- `pass_first` **không làm hại** ràng buộc hay semantic (chênh ≤ 1.3 điểm) mà mọi phim trả về đã qua kiểm tra gu (bỏ phim
  fail, phim pass lên trước). Phần +5.6 điểm semantic ở bảng 1 là nhiễu của agent.
- `pass_only` làm 20 / 60 case không còn phim nào và giảm semantic → không dùng.

## 3. Vì sao ~1/3 phim trượt ràng buộc cứng (84 / 230 phim, lượt phát lại `off`)

| nguyên nhân | phim | sửa ở đâu |
|---|---|---|
| phim ra đời sau lần rating cuối của user trong train (`max_year` tự suy ra) | 31 | `candidate_mask` chưa giới hạn theo thời điểm của user |
| thiếu genre bắt buộc: test cần **đủ tất cả** genre, `include_genres` đang lọc **bất kỳ** genre nào | 26 | `filters.candidate_mask` |
| năm < `min_year` / > `max_year` người dùng yêu cầu | 20 + 18 | tool `recommend_for_user` **không có** tham số năm; agent không truyền được |

7 case không có phim là do agent truyền tham số sai: cùng một genre ở cả `include_genres` và `exclude_genres`
(Documentary, Musical, Horror, Crime), hoặc tên genre không có trong catalog ("Science Fiction", "Family", "Sports").

Không có phim nào vi phạm "chưa rating".

## 4. Agent `deepseek/deepseek-v4.1-flash`, luồng `pass_first` (sau khi `recommend_for_user` có `min_year`/`max_year`)

`python -m scripts.run_agent_cases --verdict-filter pass_first --model openai:deepseek/deepseek-v4.1-flash --tag deepseek`
(judge vẫn `gpt-4.1-mini`). 1 case vượt giới hạn 10 bước agent ở lượt đầu, chạy lại thì qua.

| agent | case 0 phim | phim / case | lượt gọi tool | ràng buộc: phim đạt | semantic: phim đạt | semantic: hit rate | semantic: NDCG |
|---|---|---|---|---|---|---|---|
| gpt-4.1-mini (chưa có tham số năm) | 4 | 4.0 | 63 | 66.1% | 59.9% | 71.0% | 65.2% |
| **deepseek-v4.1-flash** | **0** | 5.0 | 168 | **74.3%** | **75.2%** | **93.9%** | **88.0%** |

- Lỗi "năm người dùng yêu cầu" về **0** (gpt: 34 lượt); không còn case rỗng; deepseek dùng `search_by_description` cho yêu
  cầu mô tả nhiều hơn (46 so với 18 lượt) nên semantic tăng mạnh. Hai lượt chạy khác cả model lẫn tool (tham số năm
  mới thêm), nên không tách được phần nào do model.
- Đổi lại deepseek gọi tool gần 3 lần nhiều hơn (hay gọi thêm `get_user_profile`, `explain_match`, `get_peer_opinion`).
- Còn lại: 54 lượt "phim ra đời sau lần chấm cuối của user" và 26 lượt "thiếu genre bắt buộc" — cả hai là giới hạn của
  tool, không phải của agent.

## 5. Mặc định mới: agent + judge `deepseek-v4.1-flash`, hai lỗi tool đã sửa

Sửa: `include_genres` phải có **đủ** mọi genre; không gợi ý phim ra đời sau năm (UTC) của lần chấm cuối của user trong
train (`TasteConfig.cap_year_at_last_rating`, khớp cách bộ chấm suy ra `max_year` cho 610 / 610 user). Khoá cache judge
giờ gồm cả tên model (trước đây đổi model judge sẽ dùng lại quyết định cũ).

Cùng judge deepseek, trước (lượt deepseek ở mục 4, chấm lại) và sau khi sửa tool (lượt hiện tại):

| | ràng buộc: phim đạt | ràng buộc: case đạt hết | semantic: phim đạt | semantic: hit rate | semantic: NDCG |
|---|---|---|---|---|---|
| trước sửa tool | 74.3% | 48.3% | 72.1% | 90.9% | 84.6% |
| **sau sửa tool** | **98.3%** | **98.3%** | 68.5% | 93.9% | 83.4% |

Số case có ≥ n phim đạt **cả hai** tiêu chí (ràng buộc + semantic nếu case yêu cầu), 60 case:

| | ≥ 1 | ≥ 2 | ≥ 3 | ≥ 4 | ≥ 5 |
|---|---|---|---|---|---|
| trước sửa tool | 53 (88.3%) | 48 (80.0%) | 39 (65.0%) | 31 (51.7%) | 21 (35.0%) |
| **sau sửa tool** | **57 (95.0%)** | **53 (88.3%)** | **52 (86.7%)** | **45 (75.0%)** | **36 (60.0%)** |

(bảng đầy đủ theo từng tiêu chí trong `summary.md`)

- Taste (nhãn tập test: rating ≥ 4 là thích) không đưa vào bảng: 300 phim được gợi ý, nhưng chỉ 6 phim (ở 5 / 60 case)
  là user có rating trong tập test. Trong 6 phim đó, 3 phim được chấm 4.0 (thích), 3 phim được chấm 3.0–3.5
  (bình thường). Số mẫu này quá nhỏ để kết luận gì; gu được đánh giá qua `score_candidate` trên toàn bộ test split (`evaluation/score_candidate/experiment_note.md`).

- Case duy nhất còn trượt, `horror_after_2010`, **không thể thoả**: user 40 chấm lần cuối năm 1996 nên không có phim
  ≥ 2010 hợp lệ; agent nới dần ràng buộc (5 lượt gọi tool) thay vì nói là không có.
- Semantic giảm nhẹ ở 11 case vì hai lý do: (1) giới hạn năm thu hẹp lựa chọn (vd. user 38 chấm cuối 1996, tìm phim giống
  Before Sunrise); (2) danh sách được chấm là **kết quả của lần gọi tool cuối**, mà deepseek hay gọi thêm
  `recommend_for_user` (bỏ qua mô tả) sau `search_by_description` — vd. case thriller tâm lý có Star Trek trong danh sách.
- Judge deepseek chấm semantic chặt hơn gpt-4.1-mini khoảng 3 điểm trên cùng danh sách (72.1% so với 75.2%).

## 6. Agent hiện tại (2026-09-29)

Chạy lại 60 case bằng agent hiện tại (`trusted_ai/agent.py`, tools, history đã sửa sau mục 5), cùng judge
`deepseek-v4.1-flash`, luồng `pass_first`. Lệnh như phần Tái lập; số liệu đầy đủ trong `summary.md`.

| | case 0 phim | phim / case | ràng buộc: phim đạt | ràng buộc: case đạt hết | semantic: phim đạt | semantic: hit rate | semantic: NDCG |
|---|---|---|---|---|---|---|---|
| mục 5 (sau sửa tool) | 0 | 5.0 | 98.3% | 98.3% | 68.5% | 93.9% | 83.4% |
| **agent hiện tại** | 1 | 5.2 | 95.9% | 96.6% | **78.4%** | **100.0%** | **89.3%** |

Số case có ≥ n phim đạt:

| tiêu chí | ≥ 1 | ≥ 2 | ≥ 3 | ≥ 4 | ≥ 5 |
|---|---|---|---|---|---|
| ràng buộc | 58 (96.7%) | 58 (96.7%) | 57 (95.0%) | 57 (95.0%) | 57 (95.0%) |
| semantic | 32 (97.0%) | 31 (93.9%) | 29 (87.9%) | 25 (75.8%) | 15 (45.5%) |
| **cả hai** (mục 5) | 57 (95.0%) | 53 (88.3%) | 52 (86.7%) | 45 (75.0%) | 36 (60.0%) |
| **cả hai** (hiện tại) | **58 (96.7%)** | **57 (95.0%)** | **54 (90.0%)** | **51 (85.0%)** | **41 (68.3%)** |

- Semantic tăng ~10 điểm (78.4% so với 68.5%); mọi case có yêu cầu semantic đều có ≥ 1 phim hợp.
- `similar_to_groundhog_day_no_romance`: agent vượt giới hạn 10 bước (`GraphRecursionError`) ở cả 5 lần chạy → case
  không có phim, tính là "lỗi agent". Chưa tăng `AgentConfig.max_agent_steps`.
- `horror_after_2010` vẫn không thể thoả (xem mục 5), nhưng agent giờ trả 10 phim cũ thay vì 5 (8 lượt gọi tool) nên
  kéo tỉ lệ ràng buộc xuống nhiều hơn.
- `unreliable_narrator_mystery`: 3 / 5 phim thiếu genre Mystery — agent gọi `search_by_description` mà không truyền
  `include_genres`, lỗi của agent chứ không phải tool.
- Taste: 315 phim được gợi ý, chỉ 5 phim (5 case) có rating trong tập test: 4 phim 4.0 (thích), 1 phim 3.0 — quá ít để
  kết luận.
- Câu trả lời: không câu nào trích xác suất hay ngưỡng; 1 / 59 câu nhắc "the model" (`genre_blind_spot_recommendation`).
