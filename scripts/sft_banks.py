"""Additional SFT content banks for make_sft_seed.py (kept separate so the
generator file stays readable). All content is hand-written Vietnamese with
verified or definitionally-safe answers; no web-sourced facts beyond stable
encyclopedic basics.
"""

from __future__ import annotations

# --- More stable facts (geography / science / tech basics) ---------------
FACTS_EXTRA = [
    ("Việt Nam có bao nhiêu tỉnh thành?", "Việt Nam hiện có 63 tỉnh thành, gồm 57 tỉnh và 6 thành phố trực thuộc trung ương."),
    ("Thành phố nào đông dân nhất Việt Nam?", "Thành phố Hồ Chí Minh là thành phố đông dân nhất Việt Nam."),
    ("Sông Hồng chảy qua thủ đô nào?", "Sông Hồng chảy qua Hà Nội, thủ đô của Việt Nam."),
    ("Hà Nội còn có tên gọi nào?", "Hà Nội xưa còn có tên Thăng Long, Đông Đô, Hà Nội nghĩa là 'ở trong sông'."),
    ("Việt Nam giáp biển nào?", "Việt Nam giáp Biển Đông ở phía đông."),
    ("Quốc gia nào lớn nhất thế giới?", "Nước Nga là quốc gia có diện tích lớn nhất thế giới."),
    ("Châu lục lớn nhất thế giới?", "Châu Á là châu lục có diện tích và dân số lớn nhất thế giới."),
    ("Đại dương lớn nhất là gì?", "Thái Bình Dương là đại dương lớn nhất thế giới."),
    ("Mặt Trời là gì?", "Mặt Trời là ngôi sao ở trung tâm hệ Mặt Trời, nguồn năng lượng chính của Trái đất."),
    ("Trái đất hình gì?", "Trái đất có hình cầu, hơi dẹt ở hai cực."),
    ("Vì sao có ngày và đêm?", "Ngày và đêm xuất hiện vì Trái đất tự quay quanh trục của nó mỗi 24 giờ."),
    ("Vì sao có bốn mùa?", "Bốn mùa hình thành vì trục Trái đất nghiêng khi Trái đất quay quanh Mặt Trời."),
    ("Không khí chủ yếu gồm gì?", "Không khí chủ yếu gồm nitơ (khoảng 78%) và oxy (khoảng 21%)."),
    ("Nước có công thức hóa học là gì?", "Nước có công thức hóa học H2O — hai nguyên tử hydro và một nguyên tử oxy."),
    ("Hít thở vào thì lấy gì từ không khí?", "Khi hít vào, cơ thể lấy oxy; khi thở ra, thải khí CO2."),
    ("Tiếng động truyền trong nước hay không khí nhanh hơn?", "Âm thanh truyền trong nước nhanh hơn trong không khí."),
    ("Sét là gì?", "Sét là tia lửa điện lớn xuất hiện trong mây dông hoặc giữa mây và mặt đất."),
    ("Cầu vồng xuất hiện khi nào?", "Cầu vồng xuất hiện khi ánh sáng mặt trời khúc xạ qua các hạt nước trong không khí."),
    ("Đèn LED tiết kiệm hơn đèn sợi đốt không?", "Có — đèn LED tiết kiệm điện hơn nhiều và tuổi thọ cao hơn đèn sợi đốt."),
    ("Internet là gì?", "Internet là mạng kết nối máy tính toàn cầu, cho phép trao đổi dữ liệu khắp thế giới."),
    ("Website là gì?", "Website là tập hợp trang web truy cập được qua trình duyệt bằng một địa chỉ."),
    ("Trình duyệt là gì?", "Trình duyệt là phần mềm đọc trang web, ví dụ Chrome, Firefox, Safari."),
    ("Ổ cứng dùng để làm gì?", "Ổ cứng lưu dữ liệu lâu dài — tắt máy dữ liệu vẫn còn."),
    ("RAM khác ổ cứng thế nào?", "RAM là bộ nhớ tạm, mất dữ liệu khi tắt máy; ổ cứng lưu lâu dài."),
    ("Hệ điều hành là gì?", "Hệ điều hành là phần mềm quản lý phần cứng và chạy ứng dụng, ví dụ Windows, Linux."),
    ("Wi-Fi là gì?", "Wi-Fi là công nghệ kết nối mạng không dây qua sóng radio."),
    ("Dữ liệu là gì?", "Dữ liệu là thông tin được ghi dưới dạng số để máy tính lưu và xử lý."),
    ("Mật khẩu mạnh nên như thế nào?", "Mật khẩu mạnh nên dài, trộn chữ hoa/thường, số và ký tự đặc biệt, không dùng thông tin cá nhân."),
    ("Sao lưu dữ liệu là gì?", "Sao lưu là chép dữ liệu quan trọng sang nơi khác để phòng khi mất gốc."),
    ("Mã hóa là gì?", "Mã hóa là biến dữ liệu thành dạng đọc được chỉ khi có khóa giải mã."),
]

# --- Rewrite / explain / classify (definitional tasks) -------------------
REPHRASE = [
    ("Tôi cần cái này ngay bây giờ", "Bạn ơi, bạn có thể giúp mình việc này sớm được không?"),
    ("Nói lại từ đầu!", "Bạn giải thích lại từ đầu giúp mình nhé?"),
    ("Cái này sai rồi", "Mình thấy phần này chưa đúng — mình xem lại cùng nhau nhé?"),
    ("Gửi tôi báo cáo", "Khi bạn tiện, bạn gửi giúp mình báo cáo nhé. Cảm ơn bạn!"),
    ("Đừng làm phiền tôi", "Mình xin phép xử lý việc này sau, khi bạn rảnh nhé."),
    ("Đọc giúp cái này", "Bạn xem giúp mình phần này với được không ạ?"),
    ("Nhanh lên một chút", "Bạn có thể ưu tiên việc này giúp mình được không?"),
    ("Ý kiến này dởm", "Mình có cách nhìn khác một chút — bạn nghe thử nhé?"),
]

SUMMARIZE_PAIRS = [
    ("Hà Nội là thủ đô, nổi tiếng với Hồ Gươm, Văn Miếu và ẩm thực đường phố; thành phố hơn một nghìn năm tuổi.",
     "Hà Nội — thủ đô nghìn năm tuổi, nổi tiếng với Hồ Gươm và ẩm thực."),
    ("Miko là mô hình nhỏ 38 triệu tham số, huấn luyện từ đầu trên CPU, học tiếng Việt, tiếng Anh và code.",
     "Miko — mô hình 38M tham số, học Việt/Anh/code, train trên CPU."),
    ("Việc học lập trình cần luyện tập thường xuyên, làm bài tập nhỏ mỗi ngày và không sợ sai.",
     "Học lập trình: luyện đều, bài nhỏ mỗi ngày, không sợ sai."),
    ("Uống đủ nước giúp cơ thể tỉnh táo, da khỏe và tiêu hóa tốt; nên uống khoảng 1.5-2 lít mỗi ngày.",
     "Nên uống 1.5-2 lít nước/ngày cho cơ thể khỏe."),
]

CLASSIFY_SENTIMENT = [
    ("Hôm nay thật tuyệt vời!", "tích cực"),
    ("Mình rất buồn khi nghe tin đó.", "tiêu cực"),
    ("Phim này hay lắm, nên xem!", "tích cực"),
    ("Dịch vụ tệ nhất mình từng dùng.", "tiêu cực"),
    ("Trời hôm nay có mây, nhiệt độ khoảng 25 độ.", "trung tính"),
    ("Mình rất mong được gặp lại bạn.", "tích cực"),
    ("Đơn hàng bị trễ ba ngày rồi.", "tiêu cực"),
    ("Cuốn sách gồm mười chương, khoảng ba trăm trang.", "trung tính"),
]

CLASSIFY_TOPIC = [
    ("def add(a, b): return a + b", "code"),
    ("SELECT * FROM users WHERE active = 1", "code"),
    ("Thủ đô Hà Nội rực rỡ hoa đào mùa tết.", "tin tức / địa lý"),
    ("Món phở ngon nhất là nấu xương trong nhiều giờ.", "ẩm thực"),
    ("Cầu thủ ghi bàn ở phút bấy giờ.", "thể thao"),
    ("print('Hello') in ra màn hình chữ Hello.", "code"),
    ("Học sinh chuẩn bị kỳ thi cuối kỳ vào tháng năm.", "giáo dục"),
    ("Giá xăng điều chỉnh tăng nhẹ kỳ này.", "kinh tế"),
]

REASONING_LITE = [
    ("Mèo nằm trên thùng, thùng nằm trên bàn. Vật gì thấp nhất?",
     "Bàn thấp nhất — thùng nằm trên bàn, mèo nằm trên thùng."),
    ("Hoa đỏ nhiều hơn hoa xanh; hoa xanh nhiều hơn hoa vàng. Loại nào ít nhất?",
     "Hoa vàng ít nhất."),
    ("An đi trước Bình, Bình đi trước Cường. Ai đi cuối?",
     "Cường đi cuối."),
    ("Hôm qua là thứ ba, vậy hôm nay là thứ mấy?",
     "Hôm nay là thứ tư."),
    ("Mai là thứ sáu, vậy hôm nay là thứ mấy?",
     "Hôm nay là thứ năm."),
    ("Một tá có mấy cái?",
     "Một tá có mười hai cái."),
    ("Nếu hôm nay mưa thì đường ướt. Hôm nay có mưa. Vậy đường thế nào?",
     "Đường bị ướt."),
    ("Tất cả mèo đều sợ nước. Mimi là con mèo. Mimi thế nào?",
     "Mimi sợ nước (theo giả định)."),
    ("10 chia 2 bằng mấy?",
     "10 chia 2 bằng 5."),
    ("Nửa của 8 là mấy?",
     "Nửa của 8 là 4."),
]

# --- Longer multi-turn scenarios ----------------------------------------
MULTI_SCENARIOS = [
    [
        ("Mình muốn học Python từ số 0.", "Hãy bắt đầu với biến và print. Ví dụ: x = 5 rồi print(x) in ra 5."),
        ("Vậy phép tính trong Python sao?", "Dùng toán tử quen thuộc: print(2 + 3) in ra 5; print(2 ** 10) là lũy thừa."),
        ("Cho mình ví dụ đầy đủ nhỏ.", "x = 5\ny = 3\nprint(x + y)  # in ra 8"),
        ("Cảm ơn, dễ hiểu quá!", "Rất vui! Mai thử vòng lặp for nhé — mỗi ngày một chủ đề nhỏ thôi."),
    ],
    [
        ("Hà Nội có chỗ nào chơi cuối tuần?", "Hồ Gươm đi bộ, Thăng Long Imperial Citadel, hoặc Phố cổ ẩm thực là lựa chọn phổ biến."),
        ("Vào mấy giờ Hồ Gươm đẹp nhất?", "Sáng sớm có không khí trong lành; tối cuối tuần có phố đi bộ sầm uất."),
        ("Mình nên chuẩn bị gì?", "Giày đi bộ thoải mái, nước uống và sẵn sàng nhiều người — cuối tuần rất đông."),
    ],
    [
        ("List và tuple trong Python khác gì nhau?", "List thay đổi được (x = [1, 2]), tuple cố định (y = (1, 2))."),
        ("Khi nào nên dùng tuple?", "Khi dữ liệu không nên bị sửa — ví dụ tọa độ, hằng số nhóm."),
        ("Convert list sang tuple thế nào?", "tuple([1, 2, 3]) trả về (1, 2, 3)."),
        ("Hiểu rồi, cảm ơn bạn!", "Không có gì — cứ hỏi tiếp nếu vướng nhé!"),
    ],
]
