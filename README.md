# 🎓 STUDENT PERFORMANCE FACTOR (Part 2)

## 🖥️ Machine Learning

### **📊 สิ่งที่ได้ทำมาแล้ว**
  - ✅ **Questions & Hypothesis**
  - ✅ **Data Cleansing**
  - ✅ **Exploratory Data Analysis (EDA)**
  - ✅ **Findings and Insights**
  - ✅ **Recommendation/Action and Impact**
  - ✅ **Source** https://github.com/alliedsauce/STUDENTPERFORMANCEFACTOR-DS512

---

## **🎯 SMART Objectives**

ยกระดับนักเรียนในกลุ่มคะแนนต่ำ (คะแนนสอบ < 70) จำนวน 10% ให้อยู๋ในกลุ่มคะแนนสูง (คะแนนสอบ ≥ 70) ภายในสิ้นปีการศึกษา 2568
โดยใช้ทรัพยากรที่มีอยู่ เช่น การเพิ่มการเข้าชั้นเรียนและการเข้าถึงทรัพยากรต่าง ๆ ของสถานศึกษา ซึ่งเป็นปัจจัยที่อยู่ในขอบเขตทรัพยากรที่มี

---
## **🌐 Modeling Methodology**
**🌐 การแบ่งกลุ่มตาม SMART Objectives**
 1. แบ่งกลุ่มนักเรียนเป็น 2 กลุ่ม จากข้อมูล 6,607 Records
    - กลุ่มที่มีคะแนนต่ำ (คะแนนสอบ < 70) จำนวน 4,982 คน
    - กลุ่มที่มีคะแนนสูง (คะแนนสอบ ≥ 70) จำนวน 1,625 คน
 2. ใช้โมเดล Supervised Learning แบบ Classification
    - Label: กลุ่มที่มีคะแนนต่ำ และ กลุ่มที่มีคะแนนสูง
 3. โมเดลที่เลือกใช้
    - Logistic Regression

---

## **🌐 Data Preprocessing**
 1. Target variables & feature
    - Feature:
      ปัจจัยหลัก Attendance, Hours_Studied, Previous_scores, Tutoring_Sessions
      ปัจจัยรอง Internet_Access, Motivation_Level, Family_income, Extracurricular_Activities, Parental_Involvement
    - Target: score_group

 2. จัดกลุ่ม score_group เพื่อนำไปเป็น Target
    //รูป
    
 4. Encoding
    - ทำความสะอาดข้อมูล (Data Cleaning)
    - การแปลงข้อมูลตามลำดับ (Ordinal Encoding)
      Feature: Motivation_Level, Family_income, Parental_Involvement
      มี Data เป็น Low/Medium/High แปลงเป็น Low = 0, Medium = 1, High = 2
    - การแปลงข้อมูลแบบทวิภาค (Binary Encoding)
      Feature: Internet_Access, Extracurricular_Activities
      มี Data เป็น Yes/No แปลงเป็น Yes = 1 No = 0
    - การยืนยันข้อมูลตัวเลข (Numeric Transformation)
      Feature: Attendance, Hours_Studied, Previous_scores, Tutoring_Sessions

 6. Train/Test Split: 80/20, random_state = 42    
 7. Scaling Strategies: Standard Scalar
 8. fit Model: Logistic Regression
 9. 

 10. 
    

 11. 




---
