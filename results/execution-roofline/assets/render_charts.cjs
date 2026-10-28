'use strict';
const fs=require('fs');
const path=require('path');
const report=path.dirname(__dirname);
const echarts=require(path.join(report,'_shared/js/echarts.min.js'));
const build=require('./charts.js');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'chart_data.json'),'utf8'));
const css=fs.readFileSync(path.join(__dirname,'report.css'),'utf8');
const palette={};
for(const key of ['accent','accent2','muted','ink','rule','bg2','font']){
  palette[key]=css.match(new RegExp('--'+key+':([^;}]+)'))[1];
}
const options=build(data,palette);
const font=fs.readFileSync(path.join(report,'_shared/fonts/WenQuanYi-subset.ttf')).toString('base64');
for(const [key,option] of Object.entries(options)){
  const chart=echarts.init(null,null,{renderer:'svg',ssr:true,width:1050,height:key==='roofline'?550:480});
  option.title={text:data.chart_titles[key].title,left:18,top:12,
    textStyle:{fontFamily:palette.font,fontSize:17,color:palette.ink}};
  if(option.legend?.show!==false)option.legend.top=49;
  option.grid.top=Math.max(option.grid.top,86);
  option.backgroundColor=palette.bg2;
  chart.setOption(option);
  let svg=chart.renderToSVGString();
  svg=svg.replace(/(<svg[^>]*>)/,'$1<defs><style>@font-face{font-family:WenQuanYi;src:url(data:font/ttf;base64,'+font+')}</style></defs>');
  fs.writeFileSync(path.join(__dirname,key+'.svg'),svg);
  chart.dispose();
  console.log('EXPORTED',key);
}
