package com.rupert.daylater;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Random;
import java.util.UUID;

public class MainActivity extends Activity {
    static final int ANY=0,WEEKEND=1,WEEKDAY=2,SUNNY=1,RAINY=2;
    static final String PREFS="daylater_prefs", KEY_ITEMS="items", KEY_LAST="last";
    final List<Item> items=new ArrayList<>();
    SharedPreferences prefs;
    String screen="home";

    @Override public void onCreate(Bundle b){super.onCreate(b); prefs=getSharedPreferences(PREFS,MODE_PRIVATE); load(); home();}
    @Override public void onBackPressed(){if(screen.equals("home")) super.onBackPressed(); else home();}

    int dp(int x){return Math.round(x*getResources().getDisplayMetrics().density);}
    LinearLayout base(){LinearLayout l=new LinearLayout(this);l.setOrientation(LinearLayout.VERTICAL);l.setPadding(dp(22),dp(24),dp(22),dp(40));l.setBackgroundColor(Color.rgb(247,247,242));return l;}
    void show(LinearLayout l){ScrollView s=new ScrollView(this);s.setFillViewport(true);s.addView(l,new ScrollView.LayoutParams(-1,-2));setContentView(s);}
    TextView t(String s,int size,boolean bold){TextView v=new TextView(this);v.setText(s);v.setTextSize(size);v.setTextColor(Color.rgb(23,33,27));if(bold)v.setTypeface(null,1);return v;}
    LinearLayout.LayoutParams mt(int n){LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,-2);p.topMargin=dp(n);return p;}
    Button btn(String s){Button b=new Button(this);b.setText(s);b.setTextSize(16);b.setAllCaps(false);b.setMinHeight(dp(56));return b;}
    EditText edit(String hint){EditText e=new EditText(this);e.setHint(hint);e.setTextSize(16);return e;}
    Spinner spin(String[] a){Spinner s=new Spinner(this);s.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,a));return s;}
    RadioGroup radios(String[] a){RadioGroup g=new RadioGroup(this);g.setOrientation(RadioGroup.HORIZONTAL);for(int i=0;i<a.length;i++){RadioButton r=new RadioButton(this);r.setId(View.generateViewId());r.setText(a[i]);g.addView(r,new LinearLayout.LayoutParams(0,dp(50),1));if(i==0)r.setChecked(true);}return g;}
    int idx(RadioGroup g){for(int i=0;i<g.getChildCount();i++)if(((RadioButton)g.getChildAt(i)).isChecked())return i;return 0;}
    void head(LinearLayout l,String title){Button back=btn("‹ 回首頁");back.setOnClickListener(v->home());l.addView(back);TextView h=t(title,29,true);l.addView(h,mt(8));}
    void label(LinearLayout l,String s){TextView v=t(s,15,true);l.addView(v,mt(22));}

    void home(){screen="home";LinearLayout l=base();TextView small=t("給沒動力的那一天",13,true);small.setTextColor(Color.rgb(49,95,71));l.addView(small);l.addView(t("改天",36,true),mt(4));TextView sub=t("把有動力時想到的事存起來，等條件剛好時，直接讓過去的自己替你決定。",17,false);sub.setTextColor(Color.rgb(96,112,104));l.addView(sub,mt(8));Button draw=btn("幫我挑一個");draw.setOnClickListener(v->draw());l.addView(draw,mt(30));Button add=btn("＋ 新增想做的事");add.setOnClickListener(v->add());l.addView(add,mt(12));l.addView(t(items.size()+" 個選項",24,true),mt(26));Button list=btn("查看我的清單");list.setOnClickListener(v->list());l.addView(list,mt(12));show(l);}

    void add(){screen="add";LinearLayout l=base();head(l,"我希望有一天可以……");label(l,"想做什麼？");EditText title=edit("例如：騎腳踏車、去漫畫店、學一支舞");l.addView(title);label(l,"大約需要多久？");String[] ds={"1 小時","2 小時","3 小時","4 小時","6 小時","8 小時","整天（12 小時）"};Spinner dur=spin(ds);dur.setSelection(2);l.addView(dur);label(l,"哪種日子適合？");RadioGroup day=radios(new String[]{"不限","週末","平日"});l.addView(day);label(l,"天氣條件？");RadioGroup weather=radios(new String[]{"不限","晴天","雨天"});l.addView(weather);label(l,"備註（可留白）");EditText note=edit("例如：某人推薦的書、想去的店");note.setMinLines(2);l.addView(note);Button save=btn("加入我的選項");save.setOnClickListener(v->{String name=title.getText().toString().trim();if(name.isEmpty()){title.setError("請先寫下想做的事情");return;}int[] mins={60,120,180,240,360,480,720};items.add(new Item(UUID.randomUUID().toString(),name,mins[dur.getSelectedItemPosition()],idx(day),idx(weather),note.getText().toString().trim()));save();Toast.makeText(this,"已加入",Toast.LENGTH_SHORT).show();home();});l.addView(save,mt(28));show(l);}

    void draw(){screen="draw";LinearLayout l=base();head(l,"你現在有什麼條件？");label(l,"我現在有多少時間？");String[] ds={"1 小時","2 小時","3 小時","4 小時","6 小時","8 小時","整天（12 小時）"};Spinner dur=spin(ds);dur.setSelection(2);l.addView(dur);label(l,"今天是？");RadioGroup day=radios(new String[]{"週末","平日"});Calendar c=Calendar.getInstance();int d=c.get(Calendar.DAY_OF_WEEK);((RadioButton)day.getChildAt((d==Calendar.SATURDAY||d==Calendar.SUNDAY)?0:1)).setChecked(true);l.addView(day);label(l,"現在的天氣？");RadioGroup weather=radios(new String[]{"晴天","雨天","不想管天氣"});l.addView(weather);LinearLayout result=new LinearLayout(this);result.setOrientation(LinearLayout.VERTICAL);l.addView(result,mt(24));Button go=btn("抽一個");go.setOnClickListener(v->{int[] mins={60,120,180,240,360,480,720};int available=mins[dur.getSelectedItemPosition()];int selectedDay=idx(day)==0?WEEKEND:WEEKDAY;int wi=idx(weather);int selectedWeather=wi==0?SUNNY:(wi==1?RAINY:ANY);Item x=pick(available,selectedDay,selectedWeather);result.removeAllViews();if(x==null){result.addView(t("目前沒有符合的選項",21,true));result.addView(t("放寬條件，或先新增更多想做的事。",15,false),mt(6));}else{result.addView(t("你之前留下的選項",13,true));result.addView(t(x.title,28,true),mt(8));result.addView(t(duration(x.minutes)+" · "+dayLabel(x.day)+" · "+weatherLabel(x.weather),15,false),mt(8));if(!x.note.isEmpty())result.addView(t(x.note,16,false),mt(14));}});l.addView(go,mt(16));show(l);}

    Item pick(int available,int day,int weather){List<Item> ok=new ArrayList<>();for(Item x:items)if(x.minutes<=available&&(x.day==ANY||x.day==day)&&(weather==ANY||x.weather==ANY||x.weather==weather))ok.add(x);if(ok.isEmpty())return null;String last=prefs.getString(KEY_LAST,"");if(ok.size()>1)ok.removeIf(x->x.id.equals(last));Item x=ok.get(new Random().nextInt(ok.size()));prefs.edit().putString(KEY_LAST,x.id).apply();return x;}

    void list(){screen="list";LinearLayout l=base();head(l,"你留給未來的選項");if(items.isEmpty())l.addView(t("還沒有任何選項。",16,false),mt(18));for(Item x:new ArrayList<>(items)){LinearLayout card=new LinearLayout(this);card.setOrientation(LinearLayout.VERTICAL);card.setPadding(dp(14),dp(14),dp(14),dp(14));card.addView(t(x.title,20,true));card.addView(t(duration(x.minutes)+" · "+dayLabel(x.day)+" · "+weatherLabel(x.weather),14,false),mt(4));if(!x.note.isEmpty())card.addView(t(x.note,15,false),mt(7));Button del=btn("刪除");del.setOnClickListener(v->new AlertDialog.Builder(this).setTitle("刪除這個選項？").setMessage(x.title).setNegativeButton("取消",null).setPositiveButton("刪除",(a,b)->{items.removeIf(i->i.id.equals(x.id));save();list();}).show());card.addView(del,mt(6));l.addView(card,mt(12));}show(l);}

    String duration(int m){return m>=720?"整天":(m/60)+" 小時";}
    String dayLabel(int d){return d==WEEKEND?"週末":d==WEEKDAY?"平日":"日期不限";}
    String weatherLabel(int w){return w==SUNNY?"晴天":w==RAINY?"雨天":"天氣不限";}

    void load(){items.clear();try{JSONArray a=new JSONArray(prefs.getString(KEY_ITEMS,"[]"));for(int i=0;i<a.length();i++)items.add(Item.from(a.getJSONObject(i)));}catch(Exception ignored){}}
    void save(){try{JSONArray a=new JSONArray();for(Item x:items)a.put(x.json());prefs.edit().putString(KEY_ITEMS,a.toString()).apply();}catch(Exception e){Toast.makeText(this,"儲存失敗",Toast.LENGTH_SHORT).show();}}

    static class Item{String id,title,note;int minutes,day,weather;Item(String id,String title,int minutes,int day,int weather,String note){this.id=id;this.title=title;this.minutes=minutes;this.day=day;this.weather=weather;this.note=note;}JSONObject json()throws Exception{JSONObject o=new JSONObject();o.put("id",id);o.put("title",title);o.put("minutes",minutes);o.put("day",day);o.put("weather",weather);o.put("note",note);return o;}static Item from(JSONObject o){return new Item(o.optString("id",UUID.randomUUID().toString()),o.optString("title","未命名"),o.optInt("minutes",180),o.optInt("day",0),o.optInt("weather",0),o.optString("note",""));}}
}
