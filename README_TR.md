# simulate\_api

**simulate\_api**, toplu taşıma hattı sefer planı ve doluluk tahmini yapabilen bir simülasyon API’sidir. ESHOT 800 hattı için geliştirilen bu araç, hava durumu, haftanın günü, talep çarpanı ve çalışma saat aralığı gibi parametreleri alarak dinamik sefer frekansları (headways), araç tipleri (Standard veya Körüklü) ve doluluk oranlarını hesaplar.

## Özellikler

* **Dinamik Headway Hesaplama**: Simülasyon sırasında ortalama sefer süresine ve anlık yoğunluğa göre sefer aralıklarını (headway) otomatik ayarlar.
* **Araç Tipi Seçimi**: Beklenen yük veya doluluk oranı eşiğini aşarsa daha büyük kapasiteli körüklü araçlar atanır.
* **Yolcu Akış Modeli**: Durak bazlı Poisson geliş oranları ve kitle-çekim (gravity) yaklaşımıyla iniş-biniş modellemesi.
* **CatBoost Tabanlı Seyahat Süresi Tahmini**: Gerçek zamanlı koşullar (trafik, hava durumu, okul/holiday durumu) göz önünde bulundurularak segment segment seyahat sürelerini tahmin eder.
* **Tam CORS Desteği**: Frontend uygulamalar ile güvenli ve sorunsuz entegrasyon.

## Kurulum

1. Depoyu klonlayın:

   ```bash
   git clone https://github.com/bennuramacc/simulate_api.git
   cd simulate_api
   ```

2. Gereksinimleri yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

3. Dosyaları yerleştirin:

   * `Travel Times & Dwell Times.xlsx`
   * `passenger_arrival_rates_by_stop.xlsx`
   * `DestinationProbabilities_Corrected.xlsx`
   * `segment_model.cbm`

   Hepsi proje kök dizininde olmalı.

4. Uygulamayı çalıştırın:

   ```bash
   uvicorn simulate_api:app --host 0.0.0.0 --port 8000
   ```

## Kullanım

API, `/simulate` uç noktasına **POST** istekleri alır. İstek gövdesi JSON formatında aşağıdaki alanları içerir:

```json
{
  "weather_desc": "Clear",         // "Clear", "Cloudy", "Precipitation", "Storm"
  "temp": 16.0,                    // Hava sıcaklığı (°C)
  "demand_multiplier": 1.3,        // Yolcu talep çarpanı
  "is_school_day": true,           // Okul dönemi mi?
  "is_public_holiday": false,      // Resmi tatil mi?
  "is_pandemic": false,            // Pandemi koşulu mu?
  "start": "06:00",              // Başlangıç saati (HH:MM)
  "end": "23:30"                 // Bitiş saati (HH:MM)
}
```

Başarılı bir isteğe yanıt olarak sefer planı ve doluluk detaylarını içeren bir dizi dönülür:

```json
[
  {
    "depart_time": "06:00",
    "headway": 30,
    "bus_type": "Standard",
    "capacity": 90,
    "max_occ": 54,
    "boarded": 60,
    "trip_time": 87.64,
    "load_%": 60.00
  },
  ...
]
```

| Alan         | Açıklama                            |
| ------------ | ----------------------------------- |
| depart\_time | Seferin kalkış saati (HH\:MM)       |
| headway      | Bir sonraki sefer aralığı (dakika)  |
| bus\_type    | Atanan araç tipi (Standard/Körüklü) |
| capacity     | Aracın oturma kapasitesi            |
| max\_occ     | En yüksek anlık yolcu sayısı        |
| boarded      | Toplam binen yolcu sayısı           |
| trip\_time   | Sefer süresi (dakika)               |
| load\_%      | Doluluk oranı (%)                   |

## Simülasyon Mantığı

1. **Durak Listesi Oluşturma**: `Travel Times & Dwell Times.xlsx` dosyasından seyahat ve durma sürelerini, duraklar arası mesafeyi (KM) okuyup birleştirir.
2. **Yolcu Geliş Oranları**: Her durak için saat dilimlerine göre Poisson geliş oranları, `passenger_arrival_rates_by_stop.xlsx` dosyasından okunur.
3. **Beklenen Yük Hesabı**: Belirlenen kalkış anındaki beklenen yolcu yükü, tüm hattın seyahat süreleri göz önünde bulundurularak tahmin edilir.
4. **Seyahat Süresi Tahmini**: `segment_model.cbm` CatBoost modeliyle her segment için gerçek zamanlı hava ve trafik koşullarına göre süre tahmini yapılır.
5. **Dinamik Sefer Planı**:

   * Başlangıç saatinden bitiş saatine kadar döngüde her kalkışta:

     * Beklenen yük eşiği aşılıyorsa körüklü araç seçilir.
     * `one_trip` fonksiyonuyla tek sefer simülasyonu çalıştırılır.
     * Sonuçlara göre headway, doluluk ve araç tipi güncellenir.
     * Headway, sefer süresinin ortalama süreden sapmasına göre otomatik ayarlanır.

---

Bu README, API’yi hızlıca kurmanız, test etmeniz ve simülasyon mantığını anlamanız için rehberlik sağlar. Sorularınız veya katkılarınız için lütfen **Issues** sekmesini kullanın.

