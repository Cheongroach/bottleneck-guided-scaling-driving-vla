(function () {
  'use strict';
  function build(data, p) {
    function base() {
      return {
        animation: false, color: [p.accent, p.accent2, p.muted, p.ink],
        textStyle: {fontFamily: p.font, color: p.ink, fontSize: 13},
        tooltip: {trigger: 'axis', appendToBody: true},
        grid: {left: 68, right: 28, top: 58, bottom: 66},
        legend: {top: 0, textStyle: {color: p.muted, fontFamily: p.font}},
        xAxis: {type: 'category', axisLine: {lineStyle: {color: p.rule}},
          axisLabel: {color: p.muted}, axisTick: {show: false}},
        yAxis: {type: 'value', nameTextStyle: {color: p.muted},
          axisLabel: {color: p.muted}, splitLine: {lineStyle: {color: p.rule}}}
      };
    }
    const result = {};
    const scaling = base();
    scaling.xAxis.data = data.scaling.map(r => r.world + ' GPU');
    scaling.yAxis = [Object.assign({}, scaling.yAxis, {name: '时间 ms', min: 0, max: 350}),
      {type: 'value', name: '慢步 %', min: 0, max: 100, splitLine: {show: false},
        nameTextStyle: {color: p.muted}, axisLabel: {color: p.muted, formatter: '{value}%'}}];
    scaling.grid.right = 65;
    scaling.series = [
      {name: '平均 step', type: 'bar', data: data.scaling.map(r => r.mean_ms), barMaxWidth: 65,
        label: {show: true, position: 'top', color: p.ink, formatter: r => r.value.toFixed(1)}},
      {name: '快步平均', type: 'bar', data: data.scaling.map(r => r.fast_ms), barMaxWidth: 65,
        label: {show: true, position: 'top', color: p.ink, formatter: r => r.value.toFixed(1)}},
      {name: '慢步比例', type: 'line', yAxisIndex: 1, symbol: 'diamond', symbolSize: 10,
        lineStyle: {width: 2, type: 'dashed'}, data: data.scaling.map(r => r.slow_fraction * 100),
        label: {show: true, position: 'top', color: p.muted, formatter: r => r.value.toFixed(1) + '%'}}
    ];
    result.scaling = scaling;

    const timeline = data.traces['32'].timeline;
    const lanes = ['CPU forward','CPU backward','CPU optimizer','GPU forward',
      'GPU backward','GPU optimizer','GPU NCCL gradient','GPU NCCL tail'];
    const laneColors = [p.accent+'77',p.accent2+'77',p.muted+'77',p.accent,p.accent2,p.muted,p.ink,p.ink];
    const range = base();
    range.legend = {show: false};
    range.grid = {left: 140, right: 20, top: 20, bottom: 48};
    range.tooltip = {trigger: 'item', appendToBody: true,
      formatter: x => lanes[x.value[0]] + '<br>'+x.value[1].toFixed(3)+'–'+x.value[2].toFixed(3)+' ms'};
    range.xAxis = {type: 'value', min: 0, max: Math.ceil(timeline.duration_ms), name: '相对时间 ms',
      nameLocation: 'middle', nameGap: 30, nameTextStyle: {color: p.muted},
      axisLabel: {color: p.muted}, splitLine: {lineStyle: {color: p.rule}}};
    range.yAxis = {type: 'category', inverse: true, data: lanes,
      axisLabel: {color: p.muted, fontSize: 11}, axisLine: {show: false},
      axisTick: {show: false}, splitLine: {show: true, lineStyle: {color: p.rule}}};
    range.series = [{
      type: 'custom',
      renderItem: function (params, api) {
        const y = api.value(0);
        const a = api.coord([api.value(1), y]), b = api.coord([api.value(2), y]);
        const height = api.size([0, 1])[1] * .48;
        return {type: 'rect', shape: {x:a[0],y:a[1]-height/2,width:b[0]-a[0],height:height},
          style: {fill: laneColors[y]}};
      },
      data: lanes.flatMap((name, yi) => (timeline.lanes[name] || []).map(r => [yi,r[0],r[1]]))
    }];
    result.timeline32 = range;

    const rows = data.matrices.filter(r => ['b2_q_forward','b2_kv_forward','b2_mlp_up','b2_lora_a','b2_lora_b'].includes(r.key));
    const dispatch = base();
    dispatch.xAxis.data = ['Q','KV','MLP up','LoRA A','LoRA B'];
    dispatch.yAxis.name = '时间 μs';
    dispatch.series = [
      {name:'Kernel 中位数',type:'bar',barMaxWidth:42,data:rows.map(r=>r.warm_kernel_us.p50),
        label:{show:true,position:'top',color:p.ink,formatter:r=>r.value.toFixed(1)}},
      {name:'Event 窗口均值',type:'bar',barMaxWidth:42,data:rows.map(r=>r.warm_event_mean_ms*1000),
        label:{show:true,position:'top',color:p.ink,formatter:r=>r.value.toFixed(1)}}
    ];
    result.dispatch = dispatch;

    const kernels = base();
    kernels.xAxis.data = ['B2 · CP on','B2 · CP off'];
    kernels.yAxis.name = 'Kernel / step';
    kernels.series = ['forward','backward','other'].map(phase=>({
      name:phase,type:'bar',stack:'kernel',barMaxWidth:90,
      data:['micro2_ckpton','micro2_ckptoff'].map(key=>{
        const t=data.traces[key];
        return phase==='other' ? t.phase_table.filter(r=>!['forward','backward'].includes(r.phase)).reduce((s,r)=>s+r.kernels_per_step,0):
          t.phase_table.find(r=>r.phase===phase).kernels_per_step;
      }),label:{show:phase!=='other',position:'inside',color:p.bg2,formatter:r=>r.value.toFixed(0)}
    }));
    result.kernels = kernels;

    const gc = base();
    gc.xAxis = {type:'value',min:0,max:95,name:'Step',nameLocation:'middle',nameGap:30,
      axisLabel:{color:p.muted},splitLine:{show:false}};
    gc.yAxis.name = 'Forward / GC ms';
    gc.legend.textStyle.fontSize=11;
    gc.grid.top=64;
    const gcKeys=['ckpton_gcauto','ckpton_gcdisabled','ckptoff_gcauto'];
    const gcNames=['CP on / GC auto','CP on / GC disabled','CP off / GC auto'];
    gc.series=gcKeys.map((key,i)=>({name:gcNames[i],type:'line',showSymbol:false,
      data:data.gc_steps[key].map(r=>[r.step,r.forward_ms]),lineStyle:{width:i===0?2.2:1.7}}));
    gc.series.push({name:'Generation-2 GC',type:'scatter',symbol:'diamond',symbolSize:9,
      data:data.gc_steps.ckpton_gcauto.filter(r=>r.gc_gen2).map(r=>[r.step,r.gc_ms])});
    result.gc=gc;

    const roof = base();
    const cal=data.calibration;
    roof.grid={left:72,right:26,top:62,bottom:68};
    roof.xAxis={type:'log',logBase:10,min:1,max:4000,name:'单遍逻辑计算强度 FLOP/byte',
      nameLocation:'middle',nameGap:38,nameTextStyle:{color:p.muted},
      axisLabel:{color:p.muted},splitLine:{lineStyle:{color:p.rule}},minorSplitLine:{show:false}};
    roof.yAxis={type:'log',logBase:10,min:.3,max:200,name:'逻辑 TFLOP/s',
      nameTextStyle:{color:p.muted},axisLabel:{color:p.muted},
      splitLine:{lineStyle:{color:p.rule}}};
    const reference=cal.dense_BF16_reference_TF_s, bw=cal.copy_effective_GB_s;
    const x=[1,2,4,8,16,cal.ridge_algorithm_FLOP_per_byte,64,128,256,512,1024,2048,4000];
    const chosen=data.matrices.filter(r=>['q_forward','kv_forward','mlp_up','lora_a','lora_b','dense_reference'].includes(r.op));
    roof.tooltip={trigger:'item',appendToBody:true,formatter:function(r){
      if(r.seriesIndex===0)return '经验参考线<br>'+r.value[0].toFixed(2)+' FLOP/byte<br>'+r.value[1].toFixed(2)+' TFLOP/s';
      return r.data.name+'<br>'+r.seriesName+'<br>I_alg '+r.value[0].toFixed(2)+' FLOP/byte<br>'+r.value[1].toFixed(2)+' TFLOP/s';
    }};
    roof.series=[
      {name:'经验参考',type:'line',showSymbol:false,itemStyle:{color:p.ink},lineStyle:{color:p.ink,width:2,type:'dashed'},
        data:x.map(v=>[v,Math.min(reference,bw*v/1000)])},
      {name:'Warm kernel',type:'scatter',symbolSize:9,itemStyle:{color:p.accent},
        data:chosen.map(r=>({name:r.key,value:[r.algorithm_intensity,r.warm_kernel_TF_s],
          label:{show:['b2_lora_a','b2_mlp_up'].includes(r.key),
            formatter:r.key==='b2_lora_a'?'B2 LoRA A':'B2 MLP up',
            position:r.key==='b2_lora_a'?'right':'bottom',distance:8,color:p.muted,fontSize:11}}))},
      {name:'Cold kernel',type:'scatter',symbol:'emptyCircle',symbolSize:9,itemStyle:{color:p.accent2},
        data:chosen.map(r=>({name:r.key,value:[r.algorithm_intensity,r.cold_kernel_TF_s]}))}
    ];
    result.roofline=roof;
    return result;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports=build;
    return;
  }
  const s=getComputedStyle(document.documentElement);
  const p={};
  ['accent','accent2','muted','ink','rule','bg2','font'].forEach(k=>p[k]=s.getPropertyValue('--'+k).trim());
  document.fonts.ready.then(function(){
    const options=build(window.executionChartData,p);
    Object.entries(options).forEach(function(pair){
      const el=document.getElementById('chart-'+pair[0]);
      if(!el)return;
      const chart=echarts.init(el,null,{renderer:'svg'});
      chart.setOption(pair[1]);
      window.addEventListener('resize',function(){chart.resize();});
    });
    document.documentElement.dataset.chartsReady='true';
  }).catch(function(){document.documentElement.dataset.chartsReady='failed';});
})();
