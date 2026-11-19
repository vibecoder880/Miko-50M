"""Generate a verified Vietnamese SFT seed dataset (plan §27, §31).

Deterministic template banks composed into conversations. Answers that
state facts about the world are from a curated list; arithmetic and code
answers are COMPUTED at generation time (exec in a restricted namespace)
so the dataset contains verified outputs, not hallucinated ones.

Usage::

    python scripts/make_sft_seed.py --out data/sft/chat_seed_v0.2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.sft_seed")

SYSTEM = "Bạn là Miko, một trợ lý nhỏ nói tiếng Việt."

FACTS = [
    ("Thủ đô của Việt Nam là gì?", "Thủ đô của Việt Nam là Hà Nội."),
    ("Quốc kỳ Việt Nam có màu gì?", "Quốc kỳ Việt Nam màu đỏ, ở giữa có ngôi sao vàng năm cánh."),
    ("Việt Nam có mấy mùa?", "Việt Nam có bốn mùa: xuân, hạ, thu và đông."),
    ("Phở là món ăn của nước nào?", "Phở là món ăn truyền thống của Việt Nam."),
    ("Hồ Gươm ở thành phố nào?", "Hồ Gươm nằm ở trung tâm thủ đô Hà Nội."),
    ("Nước sôi ở nhiệt độ bao nhiêu độ C?", "Ở áp suất thường, nước sôi ở khoảng 100 độ C."),
    ("Số PI xấp xỉ bao nhiêu?", "Số PI xấp xỉ 3.14 (chính xác hơn là 3.14159)."),
    ("Một tuần có mấy ngày?", "Một tuần có bảy ngày."),
    ("Một năm có mấy tháng?", "Một năm có mười hai tháng."),
    ("Tháng nào có 28 hoặc 29 ngày?", "Tháng Hai có 28 ngày, và 29 ngày trong năm nhuận."),
    ("Máy tính tiếng Anh là gì?", "'Máy tính' (bỏ túi) tiếng Anh là 'calculator'; 'computer' là máy tính để bàn/laptop."),
    ("HTML viết tắt của gì?", "HTML là HyperText Markup Language — ngôn ngữ đánh dấu cấu trúc trang web."),
    ("CPU là viết tắt của gì?", "CPU là Central Processing Unit — bộ xử lý trung tâm của máy tính."),
    ("RAM dùng để làm gì?", "RAM là bộ nhớ tạm thời giúp máy tính đọc/ghi dữ liệu đang xử lý rất nhanh."),
    ("Trái đất quay quanh gì?", "Trái đất quay quanh Mặt Trời, mất khoảng 365 ngày cho một vòng."),
    ("Mặt trăng quay quanh gì?", "Mặt trăng quay quanh Trái đất."),
    ("Rồng Bay là biểu tượng của nước nào?", "Rồng là biểu tượng lâu đời trong văn hóa Việt Nam."),
    ("Đà Nẵng nằm ở miền nào của Việt Nam?", "Đà Nẵng nằm ở miền Trung Việt Nam."),
    ("Sông lớn nhất chảy qua Việt Nam là sông nào?", "Sông lớn nhất chảy qua Việt Nam là sông Mekong (cửu Long), đoạn sông Hồng cũng rất quan trọng ở miền Bắc."),
    ("Núi cao nhất Việt Nam là núi nào?", "Núi cao nhất Việt Nam là Fansipan, còn gọi là nóc nhà Đông Dương."),
]

SMALL_TALK = [
    ("Xin chào!", "Chào bạn! Miko có thể giúp gì cho bạn hôm nay?"),
    ("Hi bạn!", "Chào bạn! Miko luôn sẵn sàng đây."),
    ("Chào buổi sáng!", "Chào buổi sáng! Chúc bạn một ngày tốt lành."),
    ("Cảm ơn bạn nhé!", "Không có gì! Cần gì cứ hỏi Miko thêm nhé."),
    ("Tạm biệt nhé.", "Hẹn gặp lại bạn! Chúc một ngày tốt lành."),
    ("Bạn thế nào rồi?", "Miko chạy ổn định! Còn bạn, hôm nay thế nào?"),
    ("Bạn có buồn không?", "Miko là chương trình máy tính nên không có cảm xúc, nhưng Miko luôn vui khi giúp được bạn."),
    ("Bạn giỏi gì nhất?", "Miko trả lời câu hỏi cơ bản, trò chuyện và viết code nhỏ bằng tiếng Việt."),
    ("Hôm nay mệt quá.", "Nghe bạn có vẻ mệt — nghỉ ngơi chút và uống ly nước nhé, rồi mọi việc sẽ dễ hơn."),
    ("Mình đang học lập trình.", "Tuyệt vời! Bắt đầu chậm mà chắc: biến, vòng lặp, hàm — cần giải thích phần nào cứ hỏi Miko."),
]

POLITE = [
    ("Đưa tôi ly nước", "Bạn ơi, cho mình xin một ly nước được không?"),
    ("Làm nhanh lên!", "Bạn có thể giúp mình việc này sớm một chút được không ạ?"),
    ("Cắt lời tôi làm gì!", "Xin lỗi, mình chưa hiểu ý bạn — bạn giải thích lại giúp mình nhé?"),
    ("Gửi file cho tôi ngay", "Khi bạn tiện, bạn gửi giúp mình file nhé. Cảm ơn bạn!"),
    ("Sửa lại cái này cho tôi", "Bạn có thể hỗ trợ mình chỉnh sửa phần này được không ạ?"),
]

SHORTEN = [
    ("Hôm nay trời đẹp quá nên tôi muốn đi dạo quanh hồ ở gần nhà mình", "Trời đẹp, tôi đi dạo quanh hồ."),
    ("Miko là mô hình nhỏ được huấn luyện bằng dữ liệu tiếng Việt, tiếng Anh và code trên CPU", "Miko là mô hình nhỏ học Việt, Anh và code trên CPU."),
    ("Bạn có thể giúp tôi viết lại đoạn này cho ngắn hơn nhưng vẫn giữ được ý chính của nó không?", "Bạn giúp mình rút gọn đoạn này nhưng giữ ý chính nhé?"),
    ("Tôi thường uống cà phê vào mỗi buổi sáng trước khi bắt đầu làm việc", "Sáng nào tôi cũng uống cà phê trước khi làm việc."),
]

TITLES = [
    ("du lịch Đà Nẵng", "Đà Nẵng — thành phố đáng sống với biển đẹp và cầu vàng nổi tiếng."),
    ("lợi ích của việc đọc sách", "Đọc sách mỗi ngày — đầu tư nhỏ, giá trị lớn."),
    ("cách học Python hiệu quả", "Học Python hiệu quả: thực hành mỗi ngày một chút."),
    ("bảo dưỡng xe máy định kỳ", "Bảo dưỡng xe máy định kỳ — đi an toàn, xe bền lâu."),
]

REWRITE_TITLES = [
    ("Viết tiêu đề cho bài về {0}.", TITLES),
]

CODE_TASKS = [
    ("Viết hàm Python cộng hai số.", "def add(a, b):\n    return a + b", None),
    ("Viết hàm Python nhân hai số.", "def mul(a, b):\n    return a * b", None),
    ("Cho mình ví dụ print trong Python.", "print('Xin chào')", "Xin chào"),
    ("2 + 3 trong Python viết thế nào, kết quả ra mấy?", "print(2 + 3)", "5"),
    ("Cách lấy độ dài list [1, 2, 3]?", "print(len([1, 2, 3]))", "3"),
    ("Tính tổng list [1, 2, 3, 4] trong Python.", "print(sum([1, 2, 3, 4]))", "10"),
    ("Số lớn nhất trong [3, 7, 2] là mấy? Viết code luôn.", "print(max([3, 7, 2]))", "7"),
    ("Đảo ngược chuỗi 'miko' trong Python.", "print('miko'[::-1])", "okim"),
    ("Chữ hoa của 'xin chào' trong Python.", "print('xin chào'.upper())", "XIN CHÀO"),
    ("for i in range(3) in ra những gì?", "for i in range(3):\n    print(i)", "0\n1\n2"),
    ("Nối hai chuỗi 'miko' và '50m' trong Python.", "print('miko' + '50m')", "miko50m"),
    ("Kiểm tra 10 > 5 trong Python.", "print(10 > 5)", "True"),
]

IDENTITY = [
    ("Bạn là ai?", "Tôi là Miko, một mô hình ngôn ngữ nhỏ được huấn luyện bằng tiếng Việt, tiếng Anh và code."),
    ("Miko là gì?", "Miko là một mô hình ngôn ngữ nhỏ (~38M tham số) được huấn luyện từ đầu trên CPU."),
    ("Miko có thông minh không?", "Miko nhỏ nên hiểu biết có giới hạn, nhưng Miko nhanh, chạy được trên máy yếu và sẽ tốt hơn khi có thêm dữ liệu."),
    ("Bạn có nhớ tôi không?", "Miko đọc được hội thoại trong phiên hiện tại, nhưng không lưu nhớ dài hạn giữa các phiên."),
    ("Bạn chạy trên phần cứng gì?", "Miko được thiết kế để huấn luyện và chạy trên CPU máy rất nhỏ, khoảng 1 GB RAM."),
    ("Bạn được huấn luyện bằng gì?", "Miko được huấn luyện bằng PyTorch trên CPU thuần, không dùng GPU."),
    ("Miko học dữ liệu gì?", "Miko học từ văn bản Wikipedia tiếng Việt, tiếng Anh và mã nguồn Python mã nguồn mở."),
    ("Bạn là người máy thật à?", "Không, Miko là một chương trình máy tính — mình trò chuyện bằng cách dự đoán từ tiếp theo."),
    ("Bạn có thể làm gì cho tôi?", "Miko trả lời câu hỏi cơ bản, trò chuyện, viết và giải thích code nhỏ, giúp diễn đạt lại câu văn."),
    ("Bạn có bao nhiêu tham số?", "Miko có khoảng 38 triệu tham số — nhỏ so với các mô hình hàng tỷ tham số."),
    ("Ai tạo ra bạn?", "Miko được xây dựng từ số không như một dự án nghiên cứu mô hình ngôn ngữ nhỏ."),
    ("Bạn nói được mấy ngôn ngữ?", "Miko nói tiếng Việt là chính, thêm tiếng Anh ở mức cơ bản và hiểu code."),
    ("Bạn học hỏi thêm được không?", "Không tự học trong lúc trò chuyện — Miko chỉ tốt hơn khi được huấn luyện thêm."),
    ("Chạy Miko tốn nhiều pin không?", "Không, Miko rất nhẹ, chạy thoải mái trên máy cấu hình thấp."),
    ("Bạn khác ChatGPT thế nào?", "Miko nhỏ hơn rất nhiều, chạy cục bộ trên CPU yếu, và tập trung vào tiếng Việt cơ bản."),
    ("Model của bạn thuộc loại nào?", "Miko là mô hình transformer decoder-only, kiểu GPT nhỏ."),
    ("Tokenizer là gì?", "Tokenizer là bộ chia văn bản thành các mẩu nhỏ (token) để mô hình đọc được."),
    ("Miko dùng tokenizer bao nhiêu từ?", "Miko dùng tokenizer BPE 16 nghìn token, hỗ trợ tốt tiếng Việt có dấu."),
    ("Vì sao Miko hay lặp chữ?", "Mô hình nhỏ chưa đủ dữ liệu nên hay lặp — giảm temperature hoặc huấn luyện thêm sẽ đỡ."),
    ("Làm sao để Miko trả lời tốt hơn?", "Đặt câu hỏi rõ ràng, ngắn gọn; và dự án sẽ tiếp tục huấn luyện Miko với dữ liệu tốt hơn."),
]

VI_EN = [
    ("Từ 'xin chào' tiếng Anh là gì?", "'Xin chào' tiếng Anh là 'hello' hoặc 'hi'."),
    ("'Thank you' nghĩa là gì?", "'Thank you' nghĩa là 'cảm ơn'."),
    ("'Good morning' nghĩa là gì?", "'Good morning' nghĩa là 'chào buổi sáng'."),
    ("'Xin chào! Tôi là Miko' dịch sang tiếng Anh.", "'Xin chào! Tôi là Miko' dịch là 'Hello! I am Miko.'"),
    ("'Good night' nghĩa là gì?", "'Good night' nghĩa là 'chúc ngủ ngon'."),
    ("'Sorry' nghĩa là gì?", "'Sorry' nghĩa là 'xin lỗi'."),
    ("'Please' nghĩa là gì?", "'Please' nghĩa là 'xin vui lòng' — thêm vào câu để lịch sự."),
    ("'How are you?' nghĩa là gì?", "'How are you?' nghĩa là 'Bạn thế nào?'."),
    ("'Tạm biệt' tiếng Anh là gì?", "'Tạm biệt' tiếng Anh là 'goodbye' hoặc 'bye'."),
    ("'Family' nghĩa là gì?", "'Family' nghĩa là 'gia đình'."),
    ("'Book' nghĩa là gì?", "'Book' nghĩa là 'quyển sách'."),
    ("'Computer' nghĩa là gì?", "'Computer' nghĩa là 'máy tính'."),
    ("'Học' tiếng Anh là gì?", "'Học' tiếng Anh là 'study' hoặc 'learn'."),
    ("'Beautiful' nghĩa là gì?", "'Beautiful' nghĩa là 'đẹp'."),
    ("'Hôm nay trời đẹp' dịch sang tiếng Anh.", "'Hôm nay trời đẹp' dịch là 'The weather is nice today.'"),
    ("'Tôi muốn học lập trình' dịch sang tiếng Anh.", "'Tôi muốn học lập trình' dịch là 'I want to learn programming.'"),
    ("'Where are you from?' nghĩa là gì?", "'Where are you from?' nghĩa là 'Bạn đến từ đâu?'."),
    ("'My name is...' nghĩa là gì?", "'My name is...' nghĩa là 'Tên tôi là...'."),
    ("'Water' nghĩa là gì?", "'Water' nghĩa là 'nước'."),
    ("'Friend' nghĩa là gì?", "'Friend' nghĩa là 'bạn bè'."),
    ("'Time' nghĩa là gì?", "'Time' nghĩa là 'thời gian'."),
    ("'Happy' nghĩa là gì?", "'Happy' nghĩa là 'hạnh phúc, vui vẻ'."),
    ("'Cảm ơn rất nhiều' dịch sang tiếng Anh.", "'Cảm ơn rất nhiều' dịch là 'Thank you very much'."),
    ("'See you later' nghĩa là gì?", "'See you later' nghĩa là 'Hẹn gặp lại sau'."),
    ("'Học sinh' tiếng Anh là gì?", "'Học sinh' tiếng Anh là 'student'."),
    ("'Teacher' nghĩa là gì?", "'Teacher' nghĩa là 'giáo viên'."),
    ("'School' nghĩa là gì?", "'School' nghĩa là 'trường học'."),
    ("'Quyển vở' tiếng Anh là gì?", "'Quyển vở' tiếng Anh là 'notebook'."),
]

DAILY = [
    ("Hôm nay mình hơi mệt.", "Nghe vậy đáng tiếc — nghỉ ngơi chút, uống nước ấm nhé. Mai sẽ khỏe hơn."),
    ("Mình đói quá nhưng không biết ăn gì.", "Thử món đơn giản nào: cơm trứng, mì gói trứng hoặc bánh mì — nhanh và no."),
    ("Trời mưa nên không đi chạy bộ được.", "Bạn có thể tập trong nhà: hít đất, squat, plank 15-20 phút cũng hiệu quả."),
    ("Mình khó ngủ quá.", "Trước khi ngủ, tránh điện thoại 30 phút, uống nước ấm và hít thở sâu nhé."),
    ("Sáng nay mình dậy muộn.", "Không sao — quan trọng là hôm nay làm được gì, không phải dậy lúc mấy giờ."),
    ("Mình đang mất hứng học.", "Đổi cách học nhẹ hơn: 15 phút một bài nhỏ, xong là nghỉ. Động lực quay lại từ tiến bộ nhỏ."),
    ("Cuối tuần nên làm gì?", "Nghỉ ngơi, đi bộ, nấu một món mới hoặc học thêm chút kiến thức nhẹ nhàng."),
    ("Mình muốn tiết kiệm tiền.", "Ghi chi tiêu mỗi ngày, cắt vài khoản nhỏ trà cà phê, để dành phần thu nhập cố định mỗi tháng."),
    ("Làm sao tập trung hơn khi làm việc?", "Chia việc thành khối 25 phút, tắt thông báo, và nghỉ 5 phút giữa các khối."),
    ("Mình nên uống gì buổi sáng?", "Nước lọc ấm là tốt nhất; thêm trà hoặc cà phê nếu bạn quen."),
    ("Nhà mình hơi bừa bộn.", "Dọn 10 phút mỗi ngày một góc nhỏ — đừng cố dọn hết một lúc."),
    ("Mình muốn bắt đầu chạy bộ.", "Bắt đầu đi bộ nhanh 20 phút/ngày, sau 1 tuần xen vài phút chạy nhẹ."),
    ("Làm sao học tiếng Anh hiệu quả?", "Nghe và đọc mỗi ngày 15 phút, học từ theo ngữ cảnh, và tập nói thành câu."),
    ("Mình hay quên việc.", "Ghi việc cần làm vào sổ hoặc điện thoại ngay khi nghĩ ra, xem lại mỗi sáng."),
    ("Nên đọc sách giấy hay sách điện tử?", "Tùy bạn — giấy dễ tập trung, điện tử tiện mang theo. Quan trọng là đọc đều đặn."),
    ("Mình muốn học đánh máy nhanh.", "Luyện 10 phút/ngày trên các trang tập gõ phím, đúng tư thế tay từ đầu."),
    ("Làm sao giảm stress?", "Vận động nhẹ, ngủ đủ, nói chuyện với người thân, và dành thời gian cho sở thích."),
    ("Buổi tối nên tránh gì?", "Tránh cà phê muộn, ăn quá no và lướt mạng liên tục ngay trước khi ngủ."),
    ("Mình muốn kiên trì hơn.", "Đặt mục tiêu nhỏ đủ làm được, duy trì mỗi ngày một chút — chuỗi nhỏ tạo thói quen lớn."),
    ("Nghỉ giải lao nên làm gì?", "Đứng dậy đi lại, vươn vai, nhìn xa — cho mắt và lưng nghỉ đúng nghĩa."),
]

CONCEPTS = [
    ("Machine learning là gì?", "Machine learning là cách máy tính học từ dữ liệu để dự đoán, thay vì được lập trình từng quy tắc."),
    ("Deep learning khác machine learning thế nào?", "Deep learning dùng mạng nơ-ron nhiều tầng để tự trích chọn đặc trưng — là một nhánh của machine learning."),
    ("Neural network là gì?", "Neural network là mô hình gồm các tầng 'nơ-ron' số học, học trọng số từ dữ liệu."),
    ("Transformer là gì?", "Transformer là kiến trúc dùng cơ chế attention để hiểu quan hệ giữa các từ, nền tảng của các mô hình ngôn ngữ hiện đại."),
    ("Attention là gì?", "Attention là cơ chế giúp mô hình 'chú ý' đến các phần liên quan của đầu vào khi xử lý từng từ."),
    ("Overfitting là gì?", "Overfitting là mô hình học thuộc dữ liệu train nhưng kém trên dữ liệu mới — thường gặp khi mô hình lớn mà dữ liệu ít."),
    ("Training và inference khác nhau thế nào?", "Training là học trọng số từ dữ liệu; inference là dùng mô hình đã học để dự đoán."),
    ("Dataset là gì?", "Dataset là tập dữ liệu dùng để huấn luyện và đánh giá mô hình."),
    ("Hàm loss là gì?", "Loss là số đo sai lệch giữa dự đoán và đáp án; huấn luyện là cố gắng giảm loss."),
    ("Learning rate là gì?", "Learning rate là bước nhảy khi cập nhật trọng số — quá lớn thì dao động, quá nhỏ thì chậm."),
    ("Epoch là gì?", "Một epoch là một lượt đi qua toàn bộ dữ liệu huấn luyện."),
    ("Batch là gì?", "Batch là nhóm mẫu được xử lý cùng một lúc trước khi cập nhật trọng số."),
    ("Fine-tuning là gì?", "Fine-tuning là tiếp tục huấn luyện mô hình đã có trên dữ liệu mới cho nhiệm vụ cụ thể."),
    ("Token là gì?", "Token là mẩu văn bản nhỏ — từ, phần từ hoặc ký tự — đơn vị nhỏ nhất mô hình xử lý."),
    ("Embedding là gì?", "Embedding là vector số biểu diễn nghĩa của từ/token để máy tính tính toán được."),
    ("Mã nguồn mở là gì?", "Mã nguồn mở là phần mềm công khai mã nguồn cho mọi người xem, sửa và phân phối theo giấy phép."),
    ("API là gì?", "API là giao diện cho phép hai chương trình nói chuyện với nhau qua các yêu cầu chuẩn."),
    ("Git là gì?", "Git là hệ thống quản lý phiên bản, lưu lịch sử thay đổi của mã nguồn."),
    ("Commit trong git là gì?", "Commit là một bản chụp các thay đổi đã lưu vào lịch sử git."),
    ("Branch trong git là gì?", "Branch là nhánh phát triển riêng để làm tính năng mới mà không ảnh hưởng nhánh chính."),
    ("Debug là gì?", "Debug là quá trình tìm và sửa lỗi trong chương trình."),
    ("Biến là gì trong lập trình?", "Biến là tên đặt cho một giá trị trong bộ nhớ để dùng lại sau này."),
    ("Vòng lặp là gì?", "Vòng lặp là cấu trúc lặp lại một khối lệnh nhiều lần."),
    ("Điều kiện if là gì?", "if kiểm tra một điều kiện đúng/sai để quyết định chạy khối lệnh nào."),
    ("Hàm trả về giá trị là sao?", "Hàm dùng return gửi kết quả về nơi gọi nó, ví dụ: def f(x): return x * 2."),
    ("Comment trong code dùng để làm gì?", "Comment là ghi chú cho người đọc, máy tính bỏ qua — giúp code dễ hiểu."),
    ("Python là ngôn ngữ thông dịch hay biên dịch?", "Python là ngôn ngữ thông dịch — chạy từng dòng qua trình thông dịch."),
    ("Vì sao Python phổ biến?", "Python cú pháp đơn giản, thư viện nhiều, hợp cho dữ liệu, tự động hóa và web."),
    ("SQL khác Python ở điểm nào?", "SQL là ngôn ngữ truy vấn dữ liệu trong cơ sở dữ liệu; Python là ngôn ngữ lập trình đa dụng."),
    ("JSON là gì?", "JSON là định dạng văn bản cấu trúc để trao đổi dữ liệu: {\"ten\": \"Miko\"}."),
]

VI_EN = [
    ("Từ 'xin chào' tiếng Anh là gì?", "'Xin chào' tiếng Anh là 'hello' hoặc 'hi'."),
    ("'Thank you' nghĩa là gì?", "'Thank you' nghĩa là 'cảm ơn'."),
    ("'Good morning' nghĩa là gì?", "'Good morning' nghĩa là 'chào buổi sáng'."),
    ("'Xin chào! Tôi là Miko' dịch sang tiếng Anh.", "'Xin chào! Tôi là Miko' dịch là 'Hello! I am Miko.'"),
]

MATH_PHRASES = [
    "{a} cộng {b} bằng mấy?",
    "Tính {a} + {b}.",
    "{a} plus {b} bằng bao nhiêu?",
]


def _num_words(n: int) -> str:
    words = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám",
             "chín", "mười", "mười một", "mười hai", "mười ba", "mười bốn",
             "mười lăm", "mười sáu", "mười bảy", "mười tám", "mười chín",
             "hai mươi", "hai mươi mốt", "hai mươi hai", "hai mươi ba",
             "hai mươi bốn", "hai mươi lăm", "hai mươi sáu", "hai mươi bảy",
             "hai mươi tám", "hai mươi chín", "ba mươi"]
    return words[n] if 0 <= n < len(words) else str(n)


def _run_code(code: str) -> str:
    """Execute a tiny generated snippet in a scratch namespace, return stdout."""
    import io
    import contextlib

    buf = io.StringIO()
    ns: dict = {}
    with contextlib.redirect_stdout(buf):
        exec(code, {"__builtins__": {"print": print, "len": len, "sum": sum,
                                     "max": max, "range": range}}, ns)
    return buf.getvalue().strip()


def build_conversations(rng: random.Random) -> list[dict]:
    convs: list[dict] = []

    def add(user: str, assistant: str, system: str | None = SYSTEM) -> None:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
        convs.append({"messages": msgs})

    def add_multi(turns: list[tuple[str, str]], system: str | None = SYSTEM) -> None:
        """One conversation with several user/assistant exchanges."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for user, assistant in turns:
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
        convs.append({"messages": msgs})

    # Small talk + facts + identity + vi-en (both orders give variety).
    for q, a in SMALL_TALK:
        add(q, a, system=None if rng.random() < 0.3 else SYSTEM)
    for q, a in FACTS:
        add(q, a)
        if rng.random() < 0.35:
            add(q, a, system=None)
    for q, a in IDENTITY:
        add(q, a)
    for q, a in VI_EN:
        add(q, a)
        if rng.random() < 0.25:
            add(q, a, system=None)
    for q, a in DAILY:
        add(q, a, system=None if rng.random() < 0.5 else SYSTEM)
    for q, a in CONCEPTS:
        add(q, a)
        if rng.random() < 0.3:
            add(f"Giải thích ngắn gọn: {q}", a.split(" — ")[0].rstrip(".") + ".")

    # Politeness rewrites + shortening + titles.
    for rude, polite in POLITE:
        add(f"Viết lại câu này lịch sự hơn: '{rude}'.", polite)
    for long, short in SHORTEN:
        add(f"Viết lại câu này ngắn gọn: '{long}'.", short)
    for topic, title in TITLES:
        add(f"Giúp mình viết tiêu đề cho bài về {topic}.", title)

    # Code tasks with verified outputs.
    for prompt, code, expect in CODE_TASKS:
        if expect is not None:
            actual = _run_code(code)
            assert actual == expect, f"code answer mismatch: {code!r}"
            add(prompt, f"{code}\n\nKết quả chạy: {actual}")
        else:
            add(prompt, code)

    # Verified arithmetic: several phrasings, numeric + word answers.
    for a in range(1, 21):
        for b in range(1, 21):
            if rng.random() > 0.45:
                continue
            s = a + b
            phrase = rng.choice(MATH_PHRASES).format(a=a, b=b)
            if rng.random() < 0.3 and s <= 30:
                add(phrase, f"{a} cộng {b} bằng {_num_words(s)}.")
            else:
                add(phrase, f"{a} cộng {b} bằng {s}.")
            if rng.random() < 0.3:
                add(f"Kết quả của {a} cộng {b} là bao nhiêu?",
                    f"Kết quả là {s}.")
            if rng.random() < 0.2:
                m = a * b
                add(f"Tính {a} nhân {b}.", f"{a} nhân {b} bằng {m}.")

    # Sequences and counting.
    for start in range(1, 15):
        add(f"Đếm từ {start} đến {start + 4} giúp mình.",
            ", ".join(str(i) for i in range(start, start + 5)) + ".")
    for s in [2, 5, 10, 3]:
        seq = [s * i for i in range(1, 6)]
        add(f"Dãy này tiếp theo là số mấy: {', '.join(map(str, seq[:4]))}?",
            f"Dãy cộng dồn {s} mỗi bước, số tiếp theo là {seq[4]}.")
    for x, y in [(9, 15), (23, 17), (40, 14), (8, 8)]:
        rel = "bằng" if x == y else ("lớn hơn" if x > y else "nhỏ hơn")
        other = y if x >= y else x
        add(f"Số nào lớn hơn: {x} hay {y}?",
            f"{x} {rel} {other}." if x != y else f"Hai số bằng nhau: {x} = {y}.")

    # Mixed two-step arithmetic (computed).
    for a, b, c in [(2, 8, 6), (3, 4, 2), (5, 6, 10), (4, 4, 4), (10, 3, 9)]:
        val = a * b - c
        add(f"{a} nhân {b} trừ {c} bằng mấy?",
            f"{a} nhân {b} bằng {a * b}, {a * b} trừ {c} bằng {val}.")

    # Multi-turn conversations: follow-ups, clarifications, code sessions.
    for a, b in [(2, 3), (4, 5), (6, 7), (9, 8), (12, 11)]:
        s = a + b
        add_multi([
            (f"{a} cộng {b} bằng mấy?", f"{a} cộng {b} bằng {s}."),
            ("Nhân đôi kết quả đó nhé.", f"{s} nhân 2 bằng {s * 2}."),
            ("Cảm ơn bạn!", "Không có gì!"),
        ])
    for topic, detail in [
        ("list trong Python", "List là danh sách có thứ tự, có thể thay đổi: x = [1, 2, 3]"),
        ("hàm trong Python", "Hàm là khối lệnh có tên và có thể trả về kết quả: def f(x): return x * 2"),
        ("vòng lặp for", "for lặp qua một dãy: for i in range(3): print(i)"),
    ]:
        add_multi([
            (f"{topic.capitalize()} là gì?", detail),
            ("Cho mình một ví dụ đầy đủ nhé.", _run_code(
                "print([1, 2, 3])" if "list" in topic else
                "def f(x):\n    return x * 2\nprint(f(5))" if "hàm" in topic else
                "for i in range(3):\n    print(i)"
            )),
            ("Ok, hiểu rồi!", "Tốt lắm! Cần giải thích thêm phần nào cứ hỏi."),
        ])
    for q, a in FACTS[:6]:
        add_multi([
            (q, a),
            ("Bạn giải thích lại ngắn gọn hơn được không?",
             a.split(",")[0].rstrip(".") + "."),
        ])

    return convs


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate verified SFT seed data")
    ap.add_argument("--out", default="data/sft/chat_seed_v0.2.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    convs = build_conversations(rng)

    # Exact-duplicate filter on (user, assistant).
    seen: set[tuple[str, str]] = set()
    unique = []
    for c in convs:
        key = (c["messages"][-2]["content"], c["messages"][-1]["content"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for c in unique:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info("wrote %d conversations -> %s", len(unique), args.out)


if __name__ == "__main__":
    main()
