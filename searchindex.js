Search.setIndex({docnames:["auto_examples/frank_wolfe/plot_fw_stepsize","auto_examples/frank_wolfe/plot_fw_vertex_overlap","auto_examples/frank_wolfe/sg_execution_times","auto_examples/index","auto_examples/plot_accelerated","auto_examples/plot_group_lasso","auto_examples/plot_jax_copt","auto_examples/plot_saga_vs_svrg","auto_examples/proximal_splitting/plot_overlapping_group_lasso","auto_examples/proximal_splitting/plot_sparse_nuclear_norm","auto_examples/proximal_splitting/plot_tv_deblurring","auto_examples/proximal_splitting/sg_execution_times","auto_examples/sg_execution_times","datasets","frank_wolfe","generated/copt.datasets.load_covtype","generated/copt.datasets.load_img1","generated/copt.datasets.load_rcv1","generated/copt.datasets.load_url","generated/copt.minimize_frank_wolfe","generated/copt.minimize_pairwise_frank_wolfe_l1","generated/copt.minimize_primal_dual","generated/copt.minimize_proximal_gradient","generated/copt.minimize_saga","generated/copt.minimize_svrg","generated/copt.minimize_three_split","generated/copt.minimize_vrtos","generated/copt.utils.GroupL1","generated/copt.utils.HuberLoss","generated/copt.utils.L1Ball","generated/copt.utils.L1Norm","generated/copt.utils.LogLoss","generated/copt.utils.SquareLoss","generated/copt.utils.TotalVariation2D","generated/copt.utils.TraceBall","generated/copt.utils.TraceNorm","incremental","index","loss_functions","proximal_gradient","proximal_splitting"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":1,sphinx:56},filenames:["auto_examples/frank_wolfe/plot_fw_stepsize.rst","auto_examples/frank_wolfe/plot_fw_vertex_overlap.rst","auto_examples/frank_wolfe/sg_execution_times.rst","auto_examples/index.rst","auto_examples/plot_accelerated.rst","auto_examples/plot_group_lasso.rst","auto_examples/plot_jax_copt.rst","auto_examples/plot_saga_vs_svrg.rst","auto_examples/proximal_splitting/plot_overlapping_group_lasso.rst","auto_examples/proximal_splitting/plot_sparse_nuclear_norm.rst","auto_examples/proximal_splitting/plot_tv_deblurring.rst","auto_examples/proximal_splitting/sg_execution_times.rst","auto_examples/sg_execution_times.rst","datasets.rst","frank_wolfe.rst","generated/copt.datasets.load_covtype.rst","generated/copt.datasets.load_img1.rst","generated/copt.datasets.load_rcv1.rst","generated/copt.datasets.load_url.rst","generated/copt.minimize_frank_wolfe.rst","generated/copt.minimize_pairwise_frank_wolfe_l1.rst","generated/copt.minimize_primal_dual.rst","generated/copt.minimize_proximal_gradient.rst","generated/copt.minimize_saga.rst","generated/copt.minimize_svrg.rst","generated/copt.minimize_three_split.rst","generated/copt.minimize_vrtos.rst","generated/copt.utils.GroupL1.rst","generated/copt.utils.HuberLoss.rst","generated/copt.utils.L1Ball.rst","generated/copt.utils.L1Norm.rst","generated/copt.utils.LogLoss.rst","generated/copt.utils.SquareLoss.rst","generated/copt.utils.TotalVariation2D.rst","generated/copt.utils.TraceBall.rst","generated/copt.utils.TraceNorm.rst","incremental.rst","index.rst","loss_functions.rst","proximal_gradient.rst","proximal_splitting.rst"],objects:{"copt.datasets":{load_covtype:[15,0,1,""],load_img1:[16,0,1,""],load_rcv1:[17,0,1,""],load_url:[18,0,1,""]},"copt.utils":{GroupL1:[27,1,1,""],HuberLoss:[28,1,1,""],L1Ball:[29,1,1,""],L1Norm:[30,1,1,""],LogLoss:[31,1,1,""],SquareLoss:[32,1,1,""],TotalVariation2D:[33,1,1,""],TraceBall:[34,1,1,""],TraceNorm:[35,1,1,""]},"copt.utils.GroupL1":{__init__:[27,2,1,""]},"copt.utils.HuberLoss":{__init__:[28,2,1,""]},"copt.utils.L1Ball":{__init__:[29,2,1,""],lmo:[29,2,1,""]},"copt.utils.L1Norm":{__init__:[30,2,1,""]},"copt.utils.LogLoss":{Hessian:[31,2,1,""],__init__:[31,2,1,""]},"copt.utils.SquareLoss":{__init__:[32,2,1,""]},"copt.utils.TotalVariation2D":{__init__:[33,2,1,""]},"copt.utils.TraceBall":{__init__:[34,2,1,""]},"copt.utils.TraceNorm":{__init__:[35,2,1,""]},copt:{minimize_frank_wolfe:[19,0,1,""],minimize_primal_dual:[21,0,1,""],minimize_proximal_gradient:[22,0,1,""],minimize_saga:[23,0,1,""],minimize_svrg:[24,0,1,""],minimize_three_split:[25,0,1,""],minimize_vrtos:[26,0,1,""]}},objnames:{"0":["py","function","Python function"],"1":["py","class","Python class"],"2":["py","method","Python method"]},objtypes:{"0":"py:function","1":"py:class","2":"py:method"},terms:{"00it":[1,5,7,8,9,10],"01e":[5,7,8,9,10],"01it":[1,8,9,10],"02e":[5,7,8,9,10],"02it":[1,5,8,9,10],"03e":[5,7,8,9,10],"03it":[1,5,8,9,10],"04e":[5,7,8,9,10],"04it":[1,5,8,9,10],"05e":[5,8,9,10],"05it":[1,5,8,10],"06e":[5,7,8,9,10],"06it":[1,5,8,9,10],"07e":[1,5,8,9,10],"07it":[1,5,8,9,10],"08e":[1,5,7,8,9,10],"08it":[1,5,8,9,10],"09e":[1,5,7,8,9,10],"09it":[1,8,9,10],"10it":[1,5,8,9,10],"11e":[5,8,9,10],"11it":[1,5,8,9,10],"12e":[5,8,9,10],"12it":[1,8,9,10],"13e":[5,8,9,10],"13it":[1,5,8,9,10],"14e":[5,8,9,10],"14it":[1,5,8,9,10],"15e":[5,8,9,10],"15it":[1,8,9,10],"16e":[5,8,9,10],"16it":[1,5,8,9,10],"17e":[5,7,8,9,10],"17it":[5,8,9,10],"18e":[5,7,8,9,10],"18it":[1,5,8,9,10],"19e":[5,7,8,9,10],"19it":[1,8,9,10],"20it":[1,5,8,9,10],"21e":[5,7,8,9,10],"21it":[1,5,8,9,10],"22e":[5,8,9,10],"22it":[1,8,9,10],"23e":[5,7,8,9,10],"23it":[1,5,8,9,10],"24e":[5,8,9,10],"24it":[1,8,9,10],"25e":[5,7,8,9,10],"25it":[1,5,8,9,10],"26e":[5,8,9,10],"26it":[8,9,10],"27e":[5,7,8,9,10],"27it":[1,5,7,8,9,10],"28e":[5,8,9,10],"28it":[8,9,10],"29e":[5,8,9,10],"29it":[1,5,8,9,10],"30it":[1,5,8,9,10],"31e":[5,7,8,9,10],"31it":[5,8,9,10],"32e":[5,7,8,9,10],"32it":[1,8,9,10],"33e":[5,7,8,9,10],"33it":[1,5,8,9,10],"34e":[5,8,9,10],"34it":[5,8,9,10],"35e":[5,7,8,9,10],"35it":[8,9,10],"35th":25,"36e":[5,7,8,9,10],"36it":[5,8,9,10],"37e":[5,8,9,10],"37it":[8,9,10],"38e":[5,7,8,9,10],"38it":[8,9,10],"39e":[5,8,9,10],"39it":[5,7,8,9],"40it":[5,8,9,10],"41e":[5,8,9,10],"41it":[1,5,8,9,10],"42e":[5,7,8,9,10],"42it":[5,8,9,10],"43e":[5,8,9,10],"43it":[1,8,9,10],"44e":[5,8,9,10],"44it":[1,8,9,10],"45e":[5,8,9,10],"45it":[1,5,8,9,10],"46e":[5,7,8,9,10],"46it":[5,8,9,10],"47e":[5,8,9,10],"47it":[1,8,10],"48e":[5,8,9,10],"48it":[1,8,9,10],"49e":[5,8,9,10],"49it":[1,5,8,10],"50it":[5,8,9,10],"51e":[5,7,8,9,10],"51it":[1,5,8,9,10],"52e":[5,8,9,10],"52it":[1,5,8,9,10],"53e":[5,7,8,9,10],"53it":[1,8,9,10],"54e":[5,7,8,9,10],"54it":[1,8,9,10],"55e":[5,7,8,9,10],"55it":[8,9,10],"56e":[5,8,9,10],"56it":[1,5,8,9,10],"57e":[5,8,9,10],"57it":[1,5,8,10],"58e":[5,7,8,9,10],"58it":[1,8,10],"59e":[5,8,9,10],"59it":[5,8,9,10],"60it":[8,9,10],"61e":[5,7,8,9,10],"61it":[5,8,9,10],"62e":[5,8,9,10],"62it":[1,8,9,10],"63e":[5,8,9,10],"63it":[1,5,8,9,10],"64e":[5,8,9,10],"64it":[5,8,9,10],"65e":[5,7,8,9,10],"65it":[5,7,8,9,10],"66e":[5,7,8,9,10],"66it":[1,8,9,10],"67e":[5,7,8,9,10],"67it":[1,5,8,9,10],"68e":[5,8,9,10],"68it":[8,9,10],"69e":[5,7,8,9,10],"69it":[1,5,8,9,10],"70it":[5,8,9,10],"71e":[5,7,8,9,10],"71it":[1,8,9,10],"72e":[5,8,9,10],"72it":[5,8,9,10],"73e":[5,8,9,10],"73it":[8,9,10],"74e":[5,8,9,10],"74it":[5,8,9,10],"75e":[5,8,9,10],"75it":[5,8,9,10],"76e":[5,7,8,9,10],"76it":[1,5,8,9,10],"77e":[5,7,8,9,10],"77it":[5,8,9,10],"78e":[5,7,8,9,10],"78it":[1,5,8,9,10],"79e":[5,7,8,9,10],"79it":[8,9,10],"80it":[5,8,9,10],"81e":[5,7,8,9,10],"81it":[5,8,9,10],"82e":[5,7,8,9,10],"82it":[7,8,9,10],"83e":[1,5,8,9,10],"83it":[1,8,9,10],"84e":[1,5,8,9,10],"84it":[5,8,9,10],"85e":[1,5,7,8,9,10],"85it":[1,8,9,10],"86e":[1,5,7,8,9,10],"86it":[1,5,8,9,10],"87e":[1,5,7,8,9,10],"87it":[1,8,9,10],"88e":[1,5,7,8,9,10],"88it":[8,9,10],"89e":[1,5,8,9,10],"89it":[5,8,9,10],"90it":[5,8,9,10],"91e":[1,5,7,8,9,10],"91it":[5,8,9,10],"92e":[1,5,7,8,9,10],"92it":[5,8,9,10],"93e":[1,5,7,8,9,10],"93it":[1,5,8,9,10],"94e":[1,5,7,8,9,10],"94it":[5,8,9,10],"95e":[1,5,8,9,10],"95it":[1,5,8,9,10],"96e":[1,5,7,8,9,10],"96it":[1,5,7,8,9,10],"97e":[1,5,7,8,9,10],"97it":[1,5,8,9,10],"98e":[1,5,7,8,9,10],"98it":[1,5,7,8,9,10],"99e":[5,8,9,10],"99it":[5,8,9,10],"boolean":[19,21,22,23,24,25,26],"break":[23,24],"case":7,"class":[27,28,29,30,31,32,33,34,35],"float":[10,19,22,23,24,25,26,27,30],"function":[0,1,4,5,6,7,14,19,21,22,23,24,25,26,29,31],"import":[0,1,4,5,6,7,8,9,10,19,21,22,23,24,25,26],"int":[9,19,21,22,23,24,25,26],"return":[5,6,8,9,10,15,17,18,19,21,22,23,24,25,26,31],"true":[1,4,5,8,9,10,17,18,21,23,24,25,26],TOS:[8,9,10],The:[4,5,7,14,19,21,22,23,24,25,26,31],These:37,Use:[4,7],With:25,__doc__:9,__init__:[27,28,29,30,31,32,33,34,35],_base:[4,7],a_i:[23,24,26],aaron:24,abov:19,abs:[9,10,25],absolut:[29,30],acceler:[3,12,22],accept:[19,21],access:[14,19,22,23,38],account:7,accur:[27,28,29,30,31,32,33,34,35],achiev:[5,8,9,10],acquir:14,adapt:[0,1,8,10,14,19,22,25],adaptive2:[0,19],adaptive3:[0,1],adato:10,advanc:[14,23,24],after:8,agg:[0,1,4,5,6,7,8,9,10],aka:[34,35],algorithm:[1,7,19,21,22,23,24,25,26],all:3,all_beta:[5,8,9,10],all_trace_l:[5,8,9,10],all_trace_ls_tim:[8,9,10],all_trace_nol:[5,8,9,10],all_trace_nols_tim:[8,9,10],all_trace_pdhg:[8,9,10],all_trace_pdhg_nol:[8,9],all_trace_pdhg_nols_tim:[8,9],all_trace_pdhg_tim:[8,9,10],alpha:[14,21,23,24,25,26,27,28,29,30,31,32,33,34,35],alreadi:[37,38],also:14,altern:[4,7,37],although:14,amir:22,anaconda3:[0,1,4,5,6,7,8,9,10],analysi:25,antonin:21,api:[23,37],append:[0,1,5,8,9,10],appendix:8,applic:[14,21,22,25],arang:[5,7,8],arg:25,argmin_:[14,19,24,26],argument:[4,7,14,19,21,22,25],armin:[14,19],arrai:[5,8,9,10,15,17,18,19,21,22,23,24,25,26],art:37,arxiv:[14,19,25,26],askari:[14,19],assum:14,astyp:[9,10],asynchron:[24,26],attribut:[19,21,22,23,24,25,26,28,31,32,34,35],augment:[23,24],auto_exampl:12,auto_examples_frank_wolf:2,auto_examples_jupyt:3,auto_examples_proximal_split:11,auto_examples_python:3,averag:[23,24],axes:[1,4,7],axi:10,b_i:[23,24,26,31],bach:24,back:6,backend:[0,1,4,5,6,7,8,9,10],backtrack:25,backtracking_factor:[22,25],ball:[14,29],barrier:[23,24],base:22,bbox_to_anchor:[5,9],beck:22,been:1,below:[19,23,24,26],beta:[5,8,9,10,21,24],between:[4,7,8],bianp:31,binari:[15,17,18],blck:9,block:[8,9,27],blur:10,blur_matrix:10,bold:[4,7],boldsymbol:14,bool:[17,18,23,24,26],bottom:[4,5,7,8,9,10],broadli:14,callabl:[19,21,22,25,31],callback:[0,1,4,5,6,7,8,9,10,19,21,22,23,24,25,26],can:[6,14,23,24,25,26,37],cannot:[0,1,4,5,6,7,8,9,10],casotto:26,categor:37,caus:[19,21,22,23,24,25,26],cb_adato:10,cb_apgd:4,cb_pdhg:[8,10],cb_pdhg_nol:8,cb_pgd:4,cb_saga:7,cb_svrg:7,cb_to:[5,8,9,10],cb_tosl:[5,8,9],cdot:32,center:[8,10],chambol:21,chang:23,check:[1,17,18],cjlin:[15,17,18],classif:[0,15,17,18],click:[0,1,4,5,6,7,8,9,10],clip:[0,1],cmap:[9,10],code:[0,1,3,4,5,6,7,8,9,10],coeffici:[21,22,25],coincid:1,com:37,combin:[3,12,14],come:31,command:37,commun:22,compact:19,comparison:[1,2,3,4,5,7,8,10,14],compart:14,composit:[21,23,24],comput:[5,6,7,8,9,10],condat:21,condit:14,confer:25,constant:[19,30],constat:27,constrain:14,constraint:0,construct:[4,5,6,7,14],contain:37,contrari:14,conveni:38,converg:[0,4,14,21,23,24,26],convex:[14,19,21,22,24],copt:[0,1,3,4,5,7,8,9,10,12,14],correl:[8,14],could:1,cov:9,covtyp:[0,1,15],cpu:6,creat:8,criterion:[19,23,24,26],csie:[15,17,18],csr:[15,17,18],current:[0,1,4,5,6,7,8,9,10,14,19,21,22,25],damek:25,data:[5,6,8,9,10,23,24,26],dataset:[0,1,4,6,7,8,24],dataset_titl:[0,1],davi:25,debug:[23,24,26],decreas:19,def:[0,1,5,6,8,9,10],defazio:24,defin:[14,19,31,32,38],delta:28,demyanov:19,densiti:5,depend:37,deploy:37,deprec:[4,7],deriv:[24,26,31,32],derivatives_logist:31,descent:[3,12,14,22,37],describ:[19,21,22,23,24,25,26],descript:[19,21,22,23,24,25,26],desir:[4,5],determin:1,dev:[4,5],develop:37,did:[4,5],diff:10,differ:[2,3,5,8,9,10,14,37],differenti:[14,19],dimension:33,direct:[2,3,14],doe:14,doi:25,domain:14,dot:[5,6,8,9,10,31],download:[0,1,3,4,5,6,7,8,9,10,15,17,18],draft:31,dt_prev:1,dual:[10,21,37],dure:1,each:[14,19,21,22,25],easi:37,easiest:37,edu:[15,17,18],elif:1,ell_1:14,els:1,emphasi:37,enter:5,enumer:[1,5,8,9,10],epsilon:9,equal:29,ergod:21,error:6,estim:[0,1,3,4,5,6,7,8,10,11,19,23,24,25,26,40],euclidean:32,evalu:[4,7,31,32],exampl:[0,1,4,5,6,7,8,9,10,22],exceed:19,execut:[1,2,11,12,19],exit:[19,21,22,23,24,25,26],experi:8,experiment:23,extra:[23,24,25,26],extrem:1,eye:9,f_deriv:[23,24,26],f_grad:[0,1,4,5,6,8,9,10,19,21,22,25],fabian:[14,19,23,24,25,26],face:10,fall:6,fals:[4,5,8,9,10,19,21,22,24,25],fast:24,fatra:26,featur:[5,8,9],few:37,fig:[1,5,8],figlegend:[5,8,9,10],figsiz:1,figur:[0,1,4,5,6,7,8,9,10],file:[2,11,12,17,18],first:[8,21],fit_transform:8,flag:[19,21,22,23,24,25,26],fmin:[4,5,7,8,9,10],follow:38,fontweight:[4,7],form:[14,19,21,22,23,24,25,26],found:[6,15,17,18],frac:[31,32],frameon:[5,8,9,10],franci:24,frank:[2,19,37],free:19,from:[1,5,6,8,9,10,21,22,23,24,25,26,31,37],fromarrai:10,full:[0,1,4,5,6,7,8,9,10,17],g_prox:10,g_t:0,galleri:[0,1,3,4,5,6,7,8,9,10],gamma:[10,14],gap:[0,19],gauthier:25,gcf:[5,8,9,10],gener:[0,1,3,4,5,6,7,8,9,10,23,24,26,37],geoffrei:[14,19],get_backend:[0,1,4,5,6,7,8,9,10],gidel:25,gisett:[0,1],git:37,github:37,given:[19,23,24,26],global:14,googl:[0,1,4,5,6,7,8,9,10],gpu:6,grad:6,gradient:[3,6,7,10,12,14,19,21,22,23,24,25,26,37],grai:10,gray_r:9,grid:[0,1,4,5,6,7,8,9,10],ground:5,ground_truth:[5,8],group:[3,11,12,22,27,40],groupl1:[5,8],guess:[19,21,22,25],gui:[0,1,4,5,6,7,8,9,10],h_lipschitz:[8,9,10,25],h_prox:10,has:24,have:[1,19,22,23,24,26,31,37],help:[27,28,29,30,31,32,33,34,35],henc:14,here:[0,1,4,5,6,7,8,9,10],hessian:31,higher:[24,26],home:[0,1,4,5,6,7,8,9,10],how:[1,6],howev:38,html:[15,17,18,31],http:[15,17,18,25,31,37],huber:28,huberloss:9,hybrid:[10,21,37],icml:[14,19],imag:[10,16],img:10,immedi:19,implement:[5,7,14,24,37],impli:1,improv:24,imshow:[9,10],increment:24,indic:[19,21,22,23,24,25,26,29],infin:29,inform:[14,23,24],initi:[19,21,22,25,27,28,29,30,31,32,33,34,35],input:[19,31],insid:21,instal:37,instead:[4,7,14],integ:19,intern:25,interpol:[9,10],involv:21,ipynb:[0,1,4,5,6,7,8,9,10],iter:[0,1,5,6,7,8,9,10,14,19,21,22,25],its:[5,19,23,25],jaggi:[14,19],jax:[3,12],journal:21,julien:[14,23,24],jupyt:[0,1,3,4,5,6,7,8,9,10],keyword:[14,19],kilian:26,known:[21,25],l1_ball:[0,1,6],l1ball:[0,1],l1norm:[6,7,9],l_t:1,label:[0,1,4,7,8,15,17,18,31],lacost:[14,23,24],lambda:[5,6,7,8,9,10],langl:19,larg:37,larger:1,lasso:[3,11,12,22,27,40],last:1,latest:37,laurent:21,learn:25,least:8,leblond:[23,24],legend:[0,1,4,7],len:[7,8,9],leq:[14,31],less:29,level:[4,5,19,21,22,23,24,25,26],lib:[0,1,4,5,6,7,8,9,10],librari:[14,37],libsvm:[15,17,18],libsvmtool:[15,17,18],like:[14,19,21,22,23,24,25],linalg:[1,9,10],line:[5,8,9,25],line_search:[8,9,10,21,25],linear:[14,19,21,29],linearli:23,lipschitz:[0,1,4,5,8,9,10,19],lipschitzian:21,list:27,lmo:[0,1,14,19,29],load:[0,1,16],load_covtyp:[0,1],load_data:[0,1],load_gisett:[0,1],load_madelon:[0,1],load_npz:10,load_rcv1:[0,1],loc:[5,9],local:[0,1,4,5,6,7,8,9,10],log:[0,4,5,6,7,8,9,10,31],logist:[4,7,31],logloss:[0,1,4,7,8],logspac:8,loss:[5,6,8,9,10,23,27,28,31,32],low:[3,11,40],lower:[8,10],machin:25,madelon:[0,1],make:[1,8],make_regress:6,mani:1,map:[23,24,26],marc:22,marker:[1,5,8,9,10],markers:[5,8,9,10],markeveri:[1,5,8,9,10],martin:[14,19],mathcal:[14,19],mathemat:[21,37],matplotlib:[0,1,4,5,6,7,8,9,10],matplotlibdeprecationwarn:[4,7],matrix:[3,8,11,15,17,18,21,40],mattia:26,max:[8,10],max_it:[1,5,7,8,9,10,19,21,22,23,24,25,26,33],max_iter_backtrack:[22,25],max_iter_l:21,max_lipschitz:7,maximum:[19,21,22,23,24,25,26],maxit:[9,10],md5:[17,18],md5_check:[17,18],member:[23,24,26],memori:[0,1,4,5,6,7,8,9,10],messag:[19,21,22,23,24,25,26],method:[7,14,19,21,23,24,25,26,27,28,29,30,31,32,33,34,35,37],might:[1,23,24,26],min:[4,5,7,8,9,10],min_:29,minim:[14,19],minimize_frank_wolf:[0,1,14],minimize_pairwise_frank_wolfe_l1:14,minimize_primal_du:[8,10],minimize_proximal_gradi:[4,5,6],minimize_saga:7,minimize_svrg:7,minimize_three_split:[8,9,10],minimize_x:[21,22,23,25],minu:[5,8,9,10],minut:[0,1,4,5,6,7,8,9,10],misc:10,model:8,modular:37,more:14,most:14,multipli:[7,9,27,30],multivariate_norm:9,n_col:[10,16],n_featur:[0,1,4,5,6,7,8,9,10,22],n_job:[24,26],n_row:[10,16],n_sampl:[0,1,4,5,6,7,8,9,10,23,24,26],ncol:[1,5,8,9,10],ndarrai:[21,23,24,26],nearest:[9,10],neg:14,negiar:[14,19],nesterov:4,net:31,neural:[14,23,24],newli:14,next:14,nip:[23,24],noisi:10,non:[0,1,4,5,6,7,8,9,10,21,24,25,38],none:[0,1,5,8,9,10,19,21,22,23,24,25,26],nonsmooth:[23,24,26],nonzero:5,norm:[1,7,23,24,26,30,32,33,34,35],note:[14,19],notebook:[0,1,3,4,5,6,7,8,9,10],npz:10,nrow:1,ntu:[15,17,18],nuclear:[34,35],number:[0,1,7,19,21,22,23,24,25,26],numpi:[0,1,4,5,6,7,8,9,10,15,17,18,37],obj_typ:[4,7],object:[5,6,7,8,9,10,14,19,21,22,23,24,25,26],onli:[15,17,18,19,21,25],onp:6,onto:34,openopt:37,oper:[5,8,10,21,22,23,25,26,37,38],opt:[23,24,26,37],optim:[4,5,7,14,19,21,22,23,24,25,26],optimizeresult:[19,21,22,23,24,25,26],optimum:[5,8,9,10],option:[19,21,22,23,24,25,26],oracl:[14,19],order:21,org:25,origin:24,other:[19,21,22,23,24,25,26],otherwis:[0,1,29],out:[0,1,4,5,6,7,8,9,10],out_img:[5,8,9,10],output:[21,22,25],over:[14,29],overlap:[2,3,11,14,40],own:38,packag:[0,1,4,5,6,7,8,9,10],pairwis:[1,37],parallel:[23,24],paramet:[5,8,9,10,14,17,18,19,21,22,23,24,25,26,27,30],parametr:23,partial_deriv:7,pass:[1,23,24,25,26],pdhg:8,pdhg_nol:8,pedregosa:[0,1,4,5,6,7,8,9,10,14,19,23,24,25,26],pen1:26,pen2:26,pen:10,penalti:27,perform:[25,31,37],pgd:5,pgd_l:5,pil:10,pip:37,plot:[0,1,4,5,6,7,8,9,10,23,24,26],plot_acceler:[4,12],plot_fw_steps:[0,2],plot_fw_vertex_overlap:[1,2],plot_group_lasso:[5,12],plot_jax_copt:[6,12],plot_nol:[5,8,9],plot_overlapping_group_lasso:[8,11],plot_pdhg:[8,10],plot_pdhg_nol:8,plot_saga_vs_svrg:[7,12],plot_sparse_nuclear_norm:[9,11],plot_to:[5,8,9,10],plot_tos_nol:10,plot_tv_deblur:[10,11],plt:[0,1,4,5,6,7,8,9,10],pock:21,point:[23,24,26],poor:1,possibl:[21,25,38],preprint:26,preprocess:8,prev_overlap:1,previou:1,primal:[10,21,37],print:[0,1,5,8,9,10,23,24,26],problem:[4,5,7,14,19,21,22,24,25,26,29],procedur:25,proceed:25,process:[14,22,23,24],product:31,program:21,progress:7,project:[15,17,18,19,34],provid:6,prox:[5,6,7,8,9,22,23,24],prox_1:[21,25,26],prox_1_arg:25,prox_2:[21,25,26],prox_factori:7,prox_tv1d_col:10,prox_tv1d_row:10,proxim:[5,21,22,23,25,37,38],proximal_gradi:[4,5],pseudo:33,pure:37,purpos:37,pylab:[0,1,4,5,6,7,8,9,10],python3:[0,1,4,5,6,7,8,9,10],python:[0,1,3,4,5,6,7,8,9,10,37],quantifi:1,rand:[4,5,7,9],randint:8,randn:[4,5,7,8,9,10],random:[4,5,6,7,8,9,10],rang:[5,8,9],rangl:19,rank:[3,11,40],rate:21,ravel:[1,9,10],rcv1:[0,1,17],reach:[4,5],readi:38,recoveri:22,reduc:[7,26,37],refer:[7,19,21,22,23,24,25,26,31],regress:[4,7],regular:[3,7,8,9,11,12,22,40],rel:[5,8,9,10],reli:14,remi:[23,24],remov:[4,7],replac:1,repres:[19,21,22,23,24,25,26],requir:14,res:[19,21,22,25],reshap:[9,10],resiz:10,respect:[0,1,24],result:[4,5,7,8,9,10,19,21,22,23,24,25,26],result_apgd:4,result_pgd:4,result_saga:7,result_svrg:7,return_gradi:[19,21,25],return_singular_vector:[9,10],revisit:[14,19],right:[0,1],routin:[14,19],row:1,rubinov:19,run:[0,1,4,5,6,7,8,9,10],runtimewarn:[4,5],s11228:25,s_t:1,saga:[3,12,23,24,26,37],same:[1,19],sampl:16,scalabl:[23,24],scale:[5,8,10,37],scatterpoint:[5,8,9,10],scheme:25,scipi:[1,5,9,10,15,17,18,19,21,22,23,24,25,26,37],script:[0,1,4,5,6,7,8,9,10],search:[5,8,9,25],second:[0,1,4,5,6,7,8,9,10],see:[19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],seed:[4,5,7,8,9,10],select:[1,14],self:[27,28,29,30,31,32,33,34,35],set:[0,8,14,19,25],set_titl:[1,5,8,9,10],set_xlabel:[1,5,8,9,10],set_xlim:[8,9,10],set_xtick:[5,8,9,10],set_ylabel:[1,5,8,9,10],set_ylim:[5,8,9,10],set_yscal:[5,8,9,10],set_ytick:[5,8,9,10],shape:[0,1,6,9,10,33,34,35],sharei:[5,8,9,10],should:[19,21,31],show:[0,1,4,5,6,7,8,9,10],sigma:[5,9,31],sigma_2:9,sigma_hat:9,sigmoid:31,sign:8,signal:22,signatur:[27,28,29,30,31,32,33,34,35],similar:37,simon:[14,23,24],sinc:1,singl:[1,21,22,25],singular:[34,35],site:[0,1,4,5,6,7,8,9,10],size:[1,2,3,5,8,9,10,14,19,22,23,24,25,26],sklearn:[6,8],slightli:[0,1],smooth:[21,25,38],snapshot:7,sol:6,solut:[19,21,22,23,24,25,26],solv:[7,14,22,23,24,25,26,29],solver:[5,8,9,10],some:[5,8,9,23,24,26],sometim:[7,14],sourc:[0,1,3,4,5,6,7,8,9,10,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],spars:[1,3,5,7,10,11,14,15,17,18,19,21,24,40],speed:[0,4],sphinx:[0,1,3,4,5,6,7,8,9,10],splinalg:[1,9,10],split:[8,10,21,25,26,37],sqrt:8,squar:[6,8,32],squareloss:[5,10],standardscal:8,start:[23,24,25,26],startswith:1,state:37,step:[1,2,3,5,8,9,10,14,19,22,23,24,25,26],step_siz:[0,1,4,5,7,8,9,10,19,21,22,23,24,25,26],step_size2:[8,10,21],stochast:[7,23,24,37],stop:[19,23,24,26],store:[0,1],strategi:[0,22],strongli:24,suboptim:[4,7],subplot:[1,5,8,9,10],subplots_adjust:[5,8,9,10],subset:17,success:[19,21,22,23,24,25,26],successfulli:[19,21,22,23,24,25,26],sum:[6,7,10,29,30,34,35],sum_:[23,24,26,31],sum_i:[14,30],support:24,svd:[9,10],svrg:[3,12,37],symptomat:1,synthet:9,system:[14,23,24],take:[14,15,17,18,19,21,22,25],teboul:22,tensorflow:6,term:[21,38],termin:[19,21,22,23,24,25,26],than:[24,26,29],thank:14,theori:21,thi:[1,6,7,8,14,15,17,18,19,21,23,24,25,26,27,29],thoma:21,thread:[24,26],three:[8,10,25,26,37],threshold:9,through:[5,23,24,26,38],tight_layout:[0,1],time:[0,1,4,5,6,7,8,9,10],titl:[0,4,7],tmp1:10,tmp2:10,tol:[1,4,5,7,8,9,10,19,21,22,23,24,25,26,33],toler:[4,5,8,9,10,19,23,24,26],tos:[8,9],tos_l:[8,9],total:[0,1,2,3,4,5,6,7,8,9,11,12,33,40],tpu:6,trace:[0,1,4,5,6,7,8,9,10,23,24,26,34,35],trace_func:[23,24,26],trace_fx:[4,6,7],trace_gt:0,trace_l:[5,8,9,10],trace_nol:[5,8,9,10],trace_pdhg:[8,10],trace_pdhg_nol:8,trace_tim:[8,9,10,23,24,26],trace_x:[5,8,9,10],tracenorm:9,track:7,train:6,truth:5,tv_prox:10,twice:1,two:[1,7],type:[19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],unlik:14,updat:[2,3,14],url:18,usag:[0,1,4,5,6,7,8,9,10],use:[1,19,22,24,26,38],used:[6,19],useful:[23,24,26],user:14,userwarn:[0,1,4,5,6,7,8,9,10],using:[0,1,4,5,6,7,8,9,10,37],usr:[0,1,4,5,6,7,8,9,10],util:[0,1,4,5,6,7,8,9,10],valu:[5,6,8,9,10,15,17,18,19,21,22,25,29,30,31,34,35],varianc:[7,26,37],variant:[1,14,19,22,23],variat:[3,11,25,33,40],vector:[1,5,19,31],verbos:[1,5,8,9,10,19,21,22,23,24,25,26],verifi:31,version:[15,17,18],vertex:[1,14],vrto:26,wai:37,warn:6,when:[19,21,31],whenev:[19,23,24,26],where:[14,19,21,22,23,25,31,32],whether:[17,18,22,23,24,25,26],which:[0,1,4,5,6,7,8,9,10,19,21,22,23,24,25,26],why:14,within:6,without:[5,8,9],wolf:[2,19,37],work:37,wotao:25,written:37,www:[15,17,18],x_i:30,x_mat:10,xla_bridg:6,xlabel:[0,4,6,7],xlim:[4,5,7,8,9,10],yin:25,ylabel:[0,4,6,7],ylim:[4,7],ymin:[4,7],you:37,your:38,yscale:[0,4,6,7],zero:[0,1,4,5,6,7,8,9,10],zip:[1,3]},titles:["Comparison of different step-sizes in Frank-Wolfe","Update Direction Overlap in Frank-Wolfe","Computation times","Examples","Accelerated gradient descent","Group Lasso regularization","Combining COPT with JAX","SAGA vs SVRG","Group lasso with overlap","Estimating a sparse and low rank matrix","Total variation regularization","Computation times","Computation times","Datasets","Frank-Wolfe and other projection-free algorithms","copt.datasets.load_covtype","copt.datasets.load_img1","copt.datasets.load_rcv1","copt.datasets.load_url","copt.minimize_frank_wolfe","copt.minimize_pairwise_frank_wolfe_l1","copt.minimize_primal_dual","copt.minimize_proximal_gradient","copt.minimize_saga","copt.minimize_svrg","copt.minimize_three_split","copt.minimize_vrtos","copt.utils.GroupL1","copt.utils.HuberLoss","copt.utils.L1Ball","copt.utils.L1Norm","copt.utils.LogLoss","copt.utils.SquareLoss","copt.utils.TotalVariation2D","copt.utils.TraceBall","copt.utils.TraceNorm","Incremental methods","Welcome to copt!","Loss functions","Gradient-based methods","Proximal splitting"],titleterms:{"function":38,acceler:4,algorithm:[14,37],base:39,combin:6,comparison:0,comput:[2,11,12],copt:[6,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,37],dataset:[13,15,16,17,18],descent:4,differ:0,direct:1,estim:9,exampl:[3,14,40],frank:[0,1,3,14],free:14,get:37,gradient:[4,39],group:[5,8],groupl1:27,huberloss:28,increment:36,jax:6,l1ball:29,l1norm:30,lasso:[5,8],load_covtyp:15,load_img1:16,load_rcv1:17,load_url:18,logloss:31,loss:38,low:9,matrix:9,method:[36,39],minimize_frank_wolf:19,minimize_pairwise_frank_wolfe_l1:20,minimize_primal_du:21,minimize_proximal_gradi:22,minimize_saga:23,minimize_svrg:24,minimize_three_split:25,minimize_vrto:26,optim:37,other:14,overlap:[1,8],pairwis:14,philosophi:37,project:14,proxim:[3,40],rank:9,refer:[8,14],regular:[5,10],saga:7,size:0,spars:9,split:[3,40],squareloss:32,start:37,step:0,svrg:7,time:[2,11,12],total:10,totalvariation2d:33,tracebal:34,tracenorm:35,updat:1,util:[27,28,29,30,31,32,33,34,35],variat:10,welcom:37,wolf:[0,1,3,14]}})